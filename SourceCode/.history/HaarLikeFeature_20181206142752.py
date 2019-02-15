import IntegralImage as II
from enum import Enum
import numpy as np
from functools import partial
from math import floor

class FeatureType(Enum):
    #pair of width and height
    TWO_VERTICAL=(1, 2)
    TWO_HORIZONTAL=(2, 1)
    THREE_VERTICAL=(1, 3)
    THREE_HORIZONTAL=(3, 1)
    FOUR=(2, 2)

class HaarLikeFeature(object):
    """
    Class representing a haar-like feature.
    """
    def __init__(self, feature_type, position, width, height, threshold, polarity):
        #Creates a new haar-like feature.
        self.type = feature_type
        self.top_left = position #row and columns
        self.bottom_right = (position[0] + height-1, position[1] + width-1)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity
        self.weight = 1
    
    def calcScore(self, ii,feature_scale=(1.0,1.0)):
        #Get score for given integral image array.
        top_left = tuple(map(floor, np.multiply(self.top_left,feature_scale)))
        bottom_right = tuple(map(floor, np.multiply(self.bottom_right,feature_scale)))
        width , height = tuple(map(floor,np.multiply((self.width,self.height),feature_scale)))
        score = 0
        if self.type == FeatureType.TWO_VERTICAL:
            first = II.sumRegion(ii, top_left, (int(top_left[0] + height / 2 - 1), bottom_right[1]))
            second = II.sumRegion(ii, (int(top_left[0] + height / 2),top_left[1] ), bottom_right)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = II.sumRegion(ii, top_left, (bottom_right[0],int(top_left[1] + width / 2-1)))
            second = II.sumRegion(ii, (top_left[0],int(top_left[1] + width / 2)), bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = II.sumRegion(ii, top_left, (bottom_right[0],int(top_left[1] + width / 3-1)))
            second = II.sumRegion(ii, (top_left[0],int(top_left[1] + width / 3)) , (bottom_right[0]
            ,int(top_left[1] + 2 * width / 3 -1)))
            third = II.sumRegion(ii, (top_left[0],int(top_left[1] + 2 * width / 3)), bottom_right)
            score = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = II.sumRegion(ii, top_left, (top_left[0]+ height / 3-1, bottom_right[1]))
            second = II.sumRegion(ii, (top_left[0]+height / 3,top_left[1]), (int(top_left[0] + 2 * height / 3-1),bottom_right[1]))
            third = II.sumRegion(ii, (int(top_left[0] + 2 * height / 3),top_left[1]), bottom_right)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            # top left area
            first = II.sumRegion(ii, top_left, (int(top_left[0] + height / 2 -1),int(top_left[1] + width / 2 -1 )))
            # top right area
            second = II.sumRegion(ii, (top_left[0],int(top_left[1] + width / 2)), (int(top_left[0] + height / 2 -1),bottom_right[1]))
            # bottom left area
            third = II.sumRegion(ii, (int(top_left[0] + height / 2),top_left[1]), (bottom_right[0],int(top_left[1] + width / 2-1)))
            # bottom right area
            fourth = II.sumRegion(ii, (int(top_left[0] + height / 2),int(top_left[1] + width / 2)), bottom_right)
            score = first + fourth - second - third 
        return score
    
    def getVotes(self,scores):
        """
        Get vote of this feature on last score
        this fucntion will alwayes be called after getScore(calc score wrapper) to assign new score first
        return 1 if this feature votes positively, otherwise 0.
        """
        return np.array(scores) > self.polarity * self.threshold 

    def getWeightedVotes(self,iis,feature_scale):
        """
        Get weighted vote of this feature on new scores given list of integral images
        return self.weight if this feature votes positively, otherwise 0.
        """
        scores = np.array(list(map(partial(self.calcScore,feature_scale=feature_scale),iis)))
        return self.weight * (self.polarity * scores >  self.polarity*self.threshold)

    def getScores(self,iis):
        """
        Calc feature score on integral image
        """
        return list(map(self.calcScore,iis))
