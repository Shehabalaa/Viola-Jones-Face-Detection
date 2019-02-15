import numpy as np
from PIL import Image
from HaarLikeFeature import FeatureType
from functools import partial
import os
import cv2
from random import shuffle
import pickle

def loadImages(path,num=-1):
    images = []
    for _file in os.listdir(path):
        if(num==0):
            break
        img_arr = np.array(cv2.imread((os.path.join(path, _file)),0))
        images.append(img_arr)
        if(num !=-1):
            num-=1
    return np.array(images)

def varianceNormalize(img):
    #img = img - img.mean()
    img_std = img.std()
    if(img_std):
        return (img/img.std())
    else:
        return img

def saveObject(obj,filename):
    f = open(filename, 'wb')
    pickle.dump(obj,f)
    f.close()

def loadObject(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj



def preProcess(image, gamma=2):
    image = cv2.blur(image,(5,5))
    #image = cv2.equalizeHist(image)
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
    image =cv2.LUT(image, table)
    return image

def meanShift(points):
    clustering = MeanShift().fit(points)
    return clustering.cluster_centers_



    
'''
def reconstruct(classifiers, img_size):
    """
    Creates an image by putting all given classifiers on top of each other
    producing an archetype of the learned class of object.
    """
    image = np.zeros(img_size)
    for c in classifiers:
        # map polarity: -1 -> 0, 1 -> 1
        polarity = pow(1 + c.polarity, 2)/4
        if c.type == FeatureType.TWO_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if y >= c.height/2:
                        sign = (sign + 1) % 2
                    image[c.top_left[1] + y, c.top_left[0] + x] += 1 * sign * c.weight
        elif c.type == FeatureType.TWO_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x >= c.width/2:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.THREE_HORIZONTAL:
            sign = polarity
            for x in range(c.width):
                if x % c.width/3 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.THREE_VERTICAL:
            for x in range(c.width):
                sign = polarity
                for y in range(c.height):
                    if x % c.height/3 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
        elif c.type == FeatureType.FOUR:
            sign = polarity
            for x in range(c.width):
                if x % c.width/2 == 0:
                    sign = (sign + 1) % 2
                for y in range(c.height):
                    if x % c.height/2 == 0:
                        sign = (sign + 1) % 2
                    image[c.top_left[0] + x, c.top_left[1] + y] += 1 * sign * c.weight
    image -= image.min()
    image /= image.max()
    image *= 255
    result = Image.fromarray(image.astype(np.uint8))
    return result
'''