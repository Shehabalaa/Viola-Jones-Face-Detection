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


def sumRegion(integral_img_arr, top_left, bottom_right):
    #Calculates the sum in the rectangle specified by the given tuples.
    row = 0
    col = 1 
    bottom_right=(int(bottom_right[0]),int(bottom_right[1]))
    if top_left == bottom_right:
        return integral_img_arr[bottom_right]
    top_right = (top_left[row], bottom_right[col])
    bottom_left = (bottom_right[row], top_left[col])
    top_left = tuple(np.subtract(top_left,(1,1)).astype(int))
    top_right = tuple(np.subtract(top_right,(1,0)).astype(int))
    bottom_left = tuple(np.subtract(bottom_left,(0,1)).astype(int))
    '''
    A B
    C D
    '''
    sumA=0.0
    sumB=0.0
    sumC=0.0
    if(top_left[row]>=0 and top_left[col]>=0):
        sumA = integral_img_arr[top_left]
    if(bottom_left[col]>=0):
        sumB = integral_img_arr[bottom_left]
    if(top_right[row]>=0):
        sumC = integral_img_arr[top_right]
    try:
        sumD = integral_img_arr[bottom_right] 
    except expression as identifier:
        print(bottom_right)
        exit()
      

    return sumD - sumC - sumB + sumA
