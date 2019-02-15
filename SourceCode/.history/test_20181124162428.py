import IntegralImage as II
from AdaBoost import StrongClassifier
from Cascade import Cascade
from Utils import loadImages
import numpy as np
import pandas as pd

import seaborn as sns
from functools import partial
import random
import pickle

min_feature_height = 1
max_feature_height = 24
min_feature_width = 1
max_feature_width = 24
train_path='../dataset/train/'
validate_path ='../dataset/validate/'
test_path='../dataset/test/'


files = [open("../iis/pos_train_iis",'rb'),open("../iis/neg_train_iis",'rb')
         ,open("../iis/pos_valid_iis",'rb'),open("../iis/neg_valid_iis",'rb')
         ,open("../iis/pos_test_iis",'rb'),open("../iis/neg_test_iis",'rb')]
pos_train_iis=pickle.load(files[0])
neg_train_iis=pickle.load(files[1])
pos_valid_iis=pickle.load(files[2])
neg_valid_iis=pickle.load(files[3])
pos_test_iis =pickle.load(files[4])
neg_test_iis =pickle.load(files[5])

for f in files:
    f.close()

# This will take a while
cascade = Cascade("./Cascade")
cascade.train(pos_train_iis,neg_train_iis,pos_valid_iis,neg_valid_iis,[.5,1.],min_feature_width, max_feature_width, min_feature_height, max_feature_height)
