import cv2
import numpy as np
def toIntegralImage(img):
    integral_img = np.zeros((img.shape[0], img.shape[1]))

    # intials first row and first colum first
    integral_img[0,0]=img[0,0]
    for c in range(1,img.shape[1]):
        integral_img[0, c] = integral_img[0, c-1] + img[0, c]
    for r in range(1,img.shape[0]):
        integral_img[r, 0] = integral_img[r-1, 0] + img[r, 0]

    for r in range(1,img.shape[0]):
        for c in range(1,img.shape[1]):
            integral_img[r, c] = integral_img[r-1, c] + integral_img[r, c-1] - integral_img[r-1, c-1]+img[r][c]
    return integral_img



def OPtoIntegralImage(img):
    integral_img = np.zeros((img.shape[0], img.shape[1]))
    
    # intials first row and first colum first
    integral_img[0, :] = np.cumsum(img[0, :])
    integral_img[:, 0] =np.cumsum(img[r, 0]
    for r in range(1,img.shape[0]):
        for c in range(1,img.shape[1]):
            integral_img[r, c] = integral_img[r-1, c] + integral_img[r, c-1] - integral_img[r-1, c-1]+img[r][c]
    return integral_img


gray = cv2.imread('faces.jpg',0)
ii1 = toIntegralImage(gray)
ii2 = toIntegralImage(gray)
res = ii1==ii2
print(np.min(res))