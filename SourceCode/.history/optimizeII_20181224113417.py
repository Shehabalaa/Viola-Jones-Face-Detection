import cv2
import numpy as np




def toIntegralImage(img):
    return cv2.integral(img)[1:,1:]
