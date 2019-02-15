from functools import partial
import numpy as np
from HaarLikeFeature import  HaarLikeFeature as HLF
from HaarLikeFeature import  FeatureType as FeatureType
import progressbar
from multiprocessing import Pool
import pickle
LOADING_BAR_LENGTH = 50

class StrongClassifier(object):
    def __init__(self,feature_list=[]):
        self.classifiers=[]
        self.features=feature_list
        self.threshold=0
        self.votes = []
        self.labels = []
        self.pos_weights=0.0
        self.iis = []
        self.chosen_features_indices=[]
        self.num_imgs=0
        self.num_features=0
    def save(self,filename):
        f = open(filename, 'wb')
        pickle.dump(self.classifiers,f)
        f.close()

    def load(self,filename):
        f = open(filename, 'rb')
        self.classifiers = pickle.load(f)
        f.close()

    def intialteLearning(self,pos_train_iis,neg_train_iis,num_classifiers,min_feature_width=1, max_feature_width=-1, min_feature_height=1, max_feature_height=-1):
        """
        Selects a set of classifiers. Iteratively takes the best classifiers based on fpr and dr as i want to reach certian false positive rate and deteion rate
        return List of selected features
        """
        num_pos = len(pos_train_iis)
        num_neg = len(neg_train_iis)
        self.num_imgs = num_pos + num_neg
        img_height, img_width = pos_train_iis[0].shape

        # Maximum feature width and height default to image width and height
        max_feature_height = img_height if max_feature_height == -1 else max_feature_height
        max_feature_width = img_width if max_feature_width == -1 else max_feature_width

        # Create initial weights and labels
        self.pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
        self.neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
        self.weights = np.hstack((self.pos_weights, self.neg_weights))
        self.labels = np.hstack((np.ones(num_pos), np.zeros(num_neg)))
        self.iis = pos_train_iis + neg_train_iis

        # Create features for all sizes and locations if no give features
        if(len(self.features)==0):
            feature_types_list=[FeatureType.TWO_VERTICAL,FeatureType.TWO_HORIZONTAL,FeatureType.THREE_HORIZONTAL,FeatureType.THREE_VERTICAL,FeatureType.FOUR]
            self.features = self.createFeatures(img_height, img_width,feature_types_list,min_feature_width, max_feature_width, min_feature_height, max_feature_height)
         
        self.num_features = len(self.features)
        self.feature_indexes = list(range(self.num_features))

        print('Calculating scores -> seletcting threshold -> getting votes for images..')
        
        scores = self.votes= np.zeros((self.num_imgs, self.num_features)).astype(np.float16) # this variable will contain scores thne contain real votes but one array used to save memory :'(
        bar = progressbar.ProgressBar() # monitor porgress
        pool = Pool(processes=None) # Use as many workers as there are CPUs

        #Calc scores
        for i in bar(range(self.num_imgs)):
            scores[i, :] = np.array(list(pool.map(partial(self.getFeatureScore, ii=self.iis[i]),self.features)))

        #Set optimal threhold for all features
        for i in bar(range(self.num_imgs)):
            pool.map(partial(self.setFeatureOptimalThreshold, scores=scores[:, i],num_pos=num_pos),self.features)

        #Get features votes
        for i in bar(range(self.num_imgs)):
            self.votes[i, :] = np.array(list(pool.map(self.getFeatureVote, self.features)))

        # select weak classifiers and save indices of selected features (weak classifiers)
        self.chosen_features_indices=[]
        self.classifiers=[]    

    def learn(self,pos_train_iis,neg_train_iis,num_classifiers,min_feature_width=1, max_feature_width=-1, min_feature_height=1, max_feature_height=-1,continue_learning=False):
        
        if(not continue_learning):
            self.intialteLearning(pos_train_iis,neg_train_iis,num_classifiers,min_feature_width, max_feature_width, min_feature_height, max_feature_height)

        print('Selecting classifiers..')
        bar = progressbar.ProgressBar() # monitor progress
        for _ in bar(range(num_classifiers)):
            classification_errors = np.zeros(len(self.feature_indexes))
            # normalize weights
            weights *= 1. / np.sum(weights)
            # select best classifier based on the weighted error
            for f in range(len(self.feature_indexes)):
                f_idx = self.feature_indexes[f]
                # classifier error is the sum of image weights where the classifier is right
                error = sum(map(lambda img_idx: weights[img_idx] if self.labels[img_idx] != self.votes[img_idx, f_idx] else 0, range(self.num_imgs)))
                classification_errors[f] = error

            # get best feature, i.e. with smallest error
            min_error_idx = np.argmin(classification_errors)
            best_error = classification_errors[min_error_idx]
            best_feature_idx = self.feature_indexes[min_error_idx]
            # set feature weight
            best_feature = self.features[best_feature_idx]
            best_feature.weight = np.log((1 - best_error) / best_error) # feature alpha (weight) and .5 is for squaring to increase feature weight 
            self.classifiers.append(best_feature) # add best feature
            self.threshold += best_feature.weight/2.

            # update image weights
            weights = np.array(list(map(lambda img_idx: weights[img_idx] if self.labels[img_idx] != self.votes[img_idx, best_feature_idx] else weights[img_idx] * best_error/(1-best_error) , range(self.num_imgs))))
            #weights = np.array(list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(self.num_imgs))))
           
            self.feature_indexes.remove(best_feature_idx) # remove feature (a feature can't be selected twice)
            self.chosen_features_indices.append(best_feature_idx) #save all chosen feature during training to remove it later as if i want to retrain(add more features to classifers ) no dupplicate happen

        self.features = [self.features[indx] for indx in range(self.num_features) if indx not in self.chosen_features_indices]
        return self.classifiers
        # there are many squares in this alogrithm that isn't in the original one this makes learning faster(accroding to an implementation guide)
    
    def setFeatureOptimalThreshold(self,feature,scores,num_pos):
        num_neg=len(scores)-num_pos
        num_pos_before=0;num_neg_before=0;err = -1;err_indx=-1
        pos_neg_flags = np.array(range(0,len(scores))) < num_pos
        scores_with_flags = sorted(zip(scores,pos_neg_flags),key=lambda x: x[0])
        print(scores_with_flags)
        for i in range(len(scores_with_flags)):
            if(scores_with_flags[i][1]): 
                num_pos_before+=1 
            else:
                num_neg_before+=1
            new_err = min(num_pos_before+num_neg - num_neg_before ,num_neg_before + num_pos-num_pos_before)
            print(num_pos_before+num_neg - num_neg_before)
            if(err_indx ==-1 or new_err < err ):
                err=new_err
                err_indx=i

        if(err_indx < len(scores)-1):
            feature.threshold = scores_with_flags[err_indx][0]+(scores_with_flags[err_indx+1][0] - scores_with_flags[err_indx][0])/2.
        else:
            feature.threshold = scores_with_flags[err_indx][0]
            
    def createFeatures(self,img_height, img_width,feature_types_list, min_feature_width, max_feature_width, min_feature_height, max_feature_height):
        print('Creating haar-like features..')
        features =[]
        for feature in feature_types_list:
            # FeatureTypes are just tuples
            feature_start_width = max(min_feature_width, feature.value[0]) if (feature!= FeatureType.THREE_HORIZONTAL) else feature.value[0]
            for feature_width in range(feature_start_width, max_feature_width, feature.value[0]):
                feature_start_height = max(min_feature_height, feature.value[1]) if (feature!= FeatureType.THREE_VERTICAL) else feature.value[1]
                for feature_height in range(feature_start_height, max_feature_height, feature.value[1]):
                    for x in range(img_width - feature_width):
                        for y in range(img_height - feature_height):
                            #print(x,y,feature_width,feature_height)
                            features.append(HLF(feature, (y,x), feature_width, feature_height, 0, 1))
                            #features.append(HLF(feature, (y,x), feature_width, feature_height, 0, -1))
        print('..done. ' + str(len(features)) + ' features created.\n')
        return features

    def getFeatureVote(self,feature):    
        return feature.getVoteLastScore()

    def getFeatureScore(self,feature, ii):    
        return feature.getNewScore(ii)

    def getFeatureWeightedVoteNewScore(self, ii,feature):    
        return feature.getWeightedVoteNewScore(ii)

    def evaluateII(self,ii):    
        result = np.sum(np.array(map(partial(self.getFeatureWeightedVoteNewScore,ii=ii),self.features))) >= self.threshold
        return result * 1 # convert bool to 1 or 0 

    def predict(self,iis):
        predictions=[]
        for ii in iis:
            predictions.append(self.evaluateII(ii))
        return np.array(predictions)


       