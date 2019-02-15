import cv2
import numpy as np




def toIntegralImage(img):
    return cv2.integral(img)[1:,1:]


gray = cv2.imread('faces.jpg',0)
ii1 = toIntegralImage(gray)
ii2 = OPtoIntegralImage(gray)
res = ii1==ii2
print(np.min(res))