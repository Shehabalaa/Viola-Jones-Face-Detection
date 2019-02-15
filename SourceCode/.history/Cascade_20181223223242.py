import AdaBoost as AB
import os
import numpy as np
from math import log
from random import shuffle
import Utils

class Cascade(object):
    def __init__(self,path="../Cascade"):
        self.stages = []
        self.stages_num=0
        self.achieved_fpr_dr = [1.0,1.0]
        self.cascade_path = path
        self.read_prev_stages(path)

    def read_prev_stages(self,path):
        try:
            for stage_file in os.listdir(path):
                stage = Utils.loadObject(path+stage_file)
                self.stages.append(stage)
                print(path +"(fpr,dr) = {1}".format(len(self.stages),stage.fpr_dr))
            self.stages_num = len(self.stages)
            if(self.stages_num>0):
                self.achived_fpr_dr = self.stages[-1].fpr_dr # actually stage fpr_dr is all cascade fpr_dr till that stage
            else:
                print("No previous stages read ...")
        except:
            print("No previous stages read ...")
            return

    def train(self,pos_train_iis, neg_train_iis, pos_valid_iis, neg_valid_iis,fpr_dr_final_goal,min_feature_width=1, max_feature_width=-1, min_feature_height=1, max_feature_height=-1):
        """
        Adding stages till reaching cascade final goal
        """
        fpr_dr_ratios = [.5,.98]
        while self.achieved_fpr_dr[0] > fpr_dr_final_goal[0]:
            """
            Adding more features(weak classifiers) to reach stage goal
            cascade_fpr_dr will start with achived goal from previous stages
            and reach stage goal ,which is determind by fpr_dr_ratio,s by adding more features to strong classifer
            note that stage goal is based on total evaluation of cascade (cascade_fpr_dr) not only current stage evaluation
            """
            #change neg validation data based every stage to focus on data fp that misclassified by cascade
            if(self.stages_num > 0):
                indices = self.predict(neg_valid_iis)
                neg_train_iis = [ neg_valid_iis[i] for i in indices[0:1000]]

            self.stages_num+=1
            print("\nTrain stage num {0}".format(self.stages_num))
            cascade_fpr_dr = self.achieved_fpr_dr
            new_stage = AB.StrongClassifier()
            self.stages.append(new_stage)
            num_weak_classifiers = 0
            new_stage_goal = np.multiply(self.achieved_fpr_dr,fpr_dr_ratios)
            continue_traning=False    
            print("To Achieve {0}".format(new_stage_goal))
            while cascade_fpr_dr[0] > new_stage_goal[0]:
                num_weak_classifiers+=int(1+ 20*log(self.stages_num,10)) #this fucntion 20*log(self.stages_num,10) just make later stages increase faster in num of features to reach goal faster
                new_stage.learn(pos_train_iis,neg_train_iis,num_weak_classifiers,min_feature_width, max_feature_width, min_feature_height, max_feature_height,continue_traning)
                thr = abs(new_stage.threshold)
                while True:
                    cascade_fpr_dr = self.evaluate(pos_valid_iis+neg_valid_iis,len(pos_valid_iis))
                    print("Print current cascsde fpr_dr{0}".format(cascade_fpr_dr))
                    if(cascade_fpr_dr[1] >= new_stage_goal[1]):
                        break
                    new_stage.threshold -= .1*thr
                    print("Stage Threshold {0}".format(new_stage.threshold ))
                continue_traning=True

            new_stage.fpr_dr=cascade_fpr_dr
            self.achieved_fpr_dr = cascade_fpr_dr
            Utils.saveObject(new_stage,self.cascade_path +"Stage"+str(self.stages_num))
            print("Achieved  fpr_dr{0} till stage {1}".format(self.achieved_fpr_dr,self.stages_num))        
            

    def predict(self,iis):
        """
        This fucntion take list of integral images
        return indices where face exist
        """
        iis=np.array(iis)
        iis_indicies = np.array(range(len(iis)))
        for stage in self.stages:
            result = stage.predict(iis[iis_indicies])
            iis_indicies = iis_indicies[result]
        return iis_indicies

    def evaluate(self,iis,num_pos):
        """
        This fucntion take list of integral images and num_pos (list is always pos_iis + neg_iis)
        return fpr and dr
        """
        iis_indicies = self.predict(iis)
        fp = np.sum(iis_indicies >= num_pos)
        dr = np.sum(iis_indicies < num_pos)
        return [fp*1./(len(iis)-num_pos),dr*1./num_pos]

        

