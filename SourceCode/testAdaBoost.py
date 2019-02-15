from functools import partial
import numpy as np
from HaarLikeFeature import  HaarLikeFeature as HLF
from HaarLikeFeature import  FeatureType as FeatureType
import progressbar
from multiprocessing import Pool
import pickle
import sys

class StrongClassifier(object):
    def __init__(self,features_types=[]):
        self.weak_classifiers=[]
        self.unselected_features=[]
        self.features_types=features_types
        self.threshold=0
        self.votes = np.array([])
        self.labels = []
        self.pos_weights=0.0
        self.iis = []
        self.num_imgs=0
        self.num_features=0
        self.fpr_dr=[0.0,0.0]
        
    def save(self,filename):
        f = open(filename, 'wb')
        pickle.dump(self.weak_classifiers,f)
        f.close()

    def load(self,filename):
        f = open(filename, 'rb')
        self.weak_classifiers = pickle.load(f)
        f.close()

    def calcScoresStep1(self):
        #Calc scores
        print('Calculating scores ...')
        pos_scores=[]
        try:
            f = open('./posScores/scores','rb')
            pos_scores = pickle.load(f)
            f.close()
        except:
            print('no pos scores available')
        count = 0
        done = 0
        print(len(pos_scores[0]))
        scores=[]
        for i in map(partial(self.getFeatureScores, iis=self.iis[len(pos_scores[0]):]),self.unselected_features):
            scores.append(i)
            count+=1
            if(done ==0 or count - done >1000 or count == self.num_features):
                done = count
                print("\rFeatures progress for calculationg score: {0}%.".format(int(100.*done/self.num_features)), end="")
                sys.stdout.flush()

        print('Calculating scores Done\n')
        scores = np.concatenate((pos_scores, scores), axis=1)
        return scores

    def optimizeParamsStep2(self,scores,num_pos):   
        #Set optimal threhold for all features
        print('Seletcting threshold ...')
        count = 0
        done = 0
        for i in map(partial(self.setFeatureOptimalThresholdandPolarity,num_pos=num_pos),self.unselected_features,scores):
            count+=1
            if(done ==0 or count - done >1000 or count == self.num_features):
                done = count
                print("\rFeatures progress for seletcting thresholds for each: {0}%.".format(int(done/self.num_features*100)), end="")
                sys.stdout.flush()
        print('Seletcting threshold  Done\n')  

    def calcVotesStep3(self,scores):
        #Get features votes
        print('Getting features votes ...')
        votes=[]
        count = 0
        done = 0
        for i in map(self.getFeatureVotes,scores,self.unselected_features):
            votes.append(i)
            count+=1
            if(done ==0 or count - done >1000 or count == self.num_features):
                done = count
                print("\rFeatures progress for Getting votes for each: {0}%.".format(int(done/self.num_features*100)), end="")
                sys.stdout.flush()            
        print('Getting votes  Done\n')
        return (np.array(votes))
        

    def initiateLearning(self,pos_train_iis,neg_train_iis,min_feature_width=1, max_feature_width=-1, min_feature_height=1, max_feature_height=-1):
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
        if(len(self.features_types)==0):
            self.features_types=[FeatureType.TWO_VERTICAL,FeatureType.TWO_HORIZONTAL,FeatureType.THREE_HORIZONTAL,FeatureType.THREE_VERTICAL,FeatureType.FOUR]
        self.unselected_features = self.createFeatures(img_height, img_width,min_feature_width, max_feature_width, min_feature_height, max_feature_height)
        
        self.num_features = len(self.unselected_features)
        self.weak_classifiers=[] 
        self.feature_indexes = list(range(self.num_features)) 

    def selectClassifires(self,num_classifiers):
        print('Selecting classifiers..')
        bar = progressbar.ProgressBar() # monitor progress
        for _ in bar(range(num_classifiers)):
            classification_errors = np.zeros(len(self.feature_indexes),dtype=np.int32)
            # normalize weights
            self.weights *= 1. / np.sum( self.weights)
            # select best classifier based on the weighted error
            #print('Calcualting errors at {0} classifier Progress ...'.format(len(self.weak_classifiers)))
            classification_errors = []
            classification_errors = list(map(lambda feature_votes: np.sum(np.multiply(self.weights,self.labels != feature_votes)),self.votes[self.feature_indexes]))
  
            # get best feature, i.e. with smallest error
            min_error_idx = np.argmin(classification_errors)
            best_error = classification_errors[min_error_idx]
            best_feature_idx = self.feature_indexes[min_error_idx]
            # set feature weight
            best_feature = self.unselected_features[best_feature_idx]
            best_feature.weight = np.log((1 - best_error) / best_error) # feature alpha (weight) and .5 is for squaring to increase feature weight 
            self.weak_classifiers.append(best_feature) # add best feature
            self.threshold += best_feature.weight/2.

            # update image weights
            self.weights = np.array(list(map(lambda img_idx: self.weights[img_idx] if self.labels[img_idx] != self.votes[best_feature_idx,img_idx] else self.weights[img_idx] * best_error/(1-best_error) , range(self.num_imgs))))
            #weights = np.array(list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(self.num_imgs))))
            self.feature_indexes.remove(best_feature_idx) # remove feature (a feature can't be selected twice)

        return self.weak_classifiers
       

    def setFeatureOptimalThresholdandPolarity(self,feature,scores,num_pos):
        num_neg=len(scores)-num_pos
        num_pos_before=0;num_neg_before=0;err = -1;err_indx=-1;num_pos_before_chosen=0
        pos_neg_flags = np.array(range(0,len(scores))) < num_pos
        scores_with_flags = sorted(zip(scores,pos_neg_flags),key=lambda x: x[0])
        #print(scores_with_flags)
        for i in range(len(scores_with_flags)):
            if(scores_with_flags[i][1]):
                num_pos_before+=1 
            else:
                num_neg_before+=1
            new_err = min(num_pos_before+num_neg - num_neg_before ,num_neg_before + num_pos-num_pos_before)
            #print(num_pos_before+num_neg - num_neg_before)
            if(err_indx ==-1 or new_err < err ):
                num_pos_before_chosen = num_pos_before
                err=new_err
                err_indx=i

        if(err_indx < len(scores)-1):
            feature.threshold = scores_with_flags[err_indx][0]+(scores_with_flags[err_indx+1][0] - scores_with_flags[err_indx][0])/2.
        else:
            feature.threshold = scores_with_flags[err_indx][0]

        if(num_pos_before_chosen<num_pos-num_pos_before_chosen):
            feature.plarity = 1
        else:
            feature.plarity = -1
            
    def createFeatures(self,img_height, img_width, min_feature_width, max_feature_width, min_feature_height, max_feature_height):
        print('Creating haar-like features..')
        features =[]
        for feature in self.features_types:
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

    def getFeatureVotes(self,scores,feature):    
        return feature.getVotes(scores)

    def getFeatureScores(self,feature, iis):    
        return feature.getScores(iis)

    def getFeatureWeightedVotes(self,feature,iis):    
        return feature.getWeightedVotes(iis)

    def predict(self,iis):
        weighted_votes = np.array(list(map(partial(self.getFeatureWeightedVotes,iis=iis),self.weak_classifiers)))
        sum_weighted_votes = np.sum(weighted_votes, axis=0) # sum evary column together (evry column is all weighted votes for each integral emage)
        return sum_weighted_votes >= self.threshold
