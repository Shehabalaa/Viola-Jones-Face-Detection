from IntegralImage import toIntegralImage as toII
import cv2
import numpy as np
import random
from sklearn.cluster import MeanShift
from Cascade import Cascade
import itertools
import Utils 
from math import floor
from functools import partial
from multiprocessing import Pool
base_detector_width = 24.

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

def nonMaxSuppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    boxes=np.array(boxes)
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")   

def detect(image,Evaluator):
    w_h_pairs=[]
    all_detected_squares = []
    w = 24 # width and height are equals as i will scan image in squares
    h = 24
    offset_w = 5
    offset_h = 5
    image_parts_ranges=[]
    image_parts_values=[]
    while(w<200 and h < image.shape[0] and w<image.shape[1]):
        r = list(range(0, image.shape[0]-h-1,int(offset_h)))
        c = list(range(0,image.shape[1]-w-1,int(offset_w)))
        image_parts_ranges += list(itertools.product(r, c))
        offset_w +=.25
        offset_h +=.25
        w = int(round(w*1.5))
        h = int(round(h*1.5))
        print(w)

    image_parts_values += list(map(lambda p: np.array(image[p[0]:p[0]+h, p[1]:p[1]+w]),image_parts_ranges))
    #for img in image_parts_values:
    #    cv2.imshow('a', img)
    #    cv2.waitKey(0)
    image_parts_values = [cv2.resize(img,(24,24)) for img in image_parts_values]
    image_parts_values_normalized = list(map(Utils.varianceNormalize,image_parts_values))
    ii_parts_values =  list(map(toII,image_parts_values_normalized))
    all_detected_squares = [(image_parts_ranges[i],image_parts_values[i].shape) for i in Evaluator.predict(ii_parts_values)]
    return all_detected_squares

'''        
def detectScaleDetector(ii,Evaluator):
    w_h_pairs=[]
    all_detected_squares = []
    w = 80 # width and height are equals as i will scan image in squares
    h = int(1.25*(w))
    offset_w = 10
    offset_h = 10
    ii_parts_ranges=[]
    ii_parts_values=[]
    while(w < ii.shape[0] and w<ii.shape[1]):
        r = list(range(0, ii.shape[0]-h,offset_h))
        c = list(range(0,ii.shape[1]-w,offset_w))
        ii_parts_ranges = list(itertools.product(r, c))
        ii_parts_values = list(map(lambda p: ii[p[0]:p[0]+h, p[1]:p[1]+w],ii_parts_ranges))
        ii_parts_values = [cv2.resize(ii,(24,24)) for ii in ii_parts_values]
        all_detected_squares += [ii_parts_ranges[i] for i in Evaluator.predict(ii_parts_values,(1,1)] #(w/24.,h/24.)
        offset_w += 1
        offset_h += 1
        if(len(all_detected_squares)):
            w_h_pairs.append((len(all_detected_squares), w,h))
        w = int(round(w*1.5))
    return all_detected_squares,w_h_pairs
'''

def main():
    Evaluator = Cascade('../Cascade/')
    #cap = cv2.VideoCapture(0)
    #while(True):
    # Capture frame-by-frame
    #ret,frame = cap.read()
    frame = cv2.imread("faces.jpg")
    frame = cv2.resize(frame,(600,400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',gray)
    #cv2.waitKey(0);
    print("ssssssssssstttttttttaaaaaaaaarttttt")
    #gray = cv2.blur(gray,(5,5))
    recs = detect(gray,Evaluator)
    #recs,w_h_pairs = detectFast(toII(Utils.varianceNormalize(gray)),Evaluator)
    recs = [[recs[i][0][1],recs[i][0][1]+recs[i][1][1],recs[i][0][0],recs[i][0][0]+recs[i][1][0]] for i in range(len(recs))]
    recs = nonMaxSuppression(recs,.1)
    [cv2.rectangle(frame,(rec[0],rec[1]),(rec[2],rec[3]), (255, 0, 0), 2) for rec in recs ]
    cv2.imshow('frame',frame)
    cv2.waitKey(0)                                  
    #cap.release()
    #cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()


"""
    Take raw frame before any previos processiong just in gray lvl
    return hand's postion as x,y,w,h
      
    w=30
    h=30
    doubts=[]
    imagnge(len(res)):
        if(res[i]==1):
                doubts.append(pos_of_images_to_detect[i])
                doubts2.append(rec_of_images_to_detect[i])
    print("Num of Scanned:{0}\nNum of TP:{1}\nNum of FN:{2}\n ".format(len(res),sum(res),len(res)-sum(res)))
    return doubts,nonMaxSuppression(doubts2,0.1) 
    #return nonMaxSuppression(doubts,0.1)
    '''
    true_point=(0,0)
    true_point_doubts=0
    for x in range(0,gray.shape[0],40):
            for y in range(0,gray.shape[1],40):
                tmp_point_doubts=0
                for doubt in doubts:
                    if(doubt[2]>=x>=doubt[0] and doubt[3]>=y>=doubt[1]):
                        tmp_point_doubts+=1
                if(tmp_point_doubts>true_point_doubts):
                    true_point=(y,x)
                    true_point_doubts=tmp_point_doubts
    return true_point
    '''es_to_detect=[]
    pos_of_images_to_detect=[]
    rec_of_images_to_detect=[]
    while(True):
        if(w >=gray.shape[0]):
            break
        w=int(w*2)
        h=int(h*2)
        for r in range(0,gray.shape[0]-h+1,15):
            for c in range(0,gray.shape[1]-w+1,15):
                #TODO scalling feature instead of resising image
                new = cv2.resize(gray[r:r+h,c:c+w],(28,28))
                #new = preProcess(new,1.2)
                #cv2.imshow('new',new)
                #cv2.waitKey(0)
                images_to_detect.append(new)
                rec_of_images_to_detect.append((c,r,c+w,r+w)) #append postions not as row and colums
                pos_of_images_to_detect.append((int(c+w/2),int(r+w/2))) #append postions not as row and colums
    
    images_ii_to_detect = list(map(toII, images_to_detect))
    res = sc.predict(images_ii_to_detect)
    doubts2=[]
""" 
