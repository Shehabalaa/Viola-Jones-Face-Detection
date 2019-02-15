from IntegralImage import toIntegralImage as toII
import cv2
import numpy as np
import random
from Cascade import Cascade
import itertools 
from math import ceil
from functools import partial
from Utils import *
base_detector_width = 24.


def detect(image,Evaluator,w=80,offset_w=20,offset_h=20):
    all_detected_squares = []
    image_parts_ranges=[]
    image_parts_values=[]
    h = int(ceil(1.1=w))
    while(h < image.shape[0] and w<image.shape[1]):
        r = list(range(0, image.shape[0]-h-1,int(offset_h)))
        c = list(range(0,image.shape[1]-w-1,int(offset_w)))
        new_range =  list(itertools.product(r, c))
        image_parts_ranges += list(itertools.product(r, c))
        image_parts_values += list(map(lambda p: np.array(image[p[0]:p[0]+h, p[1]:p[1]+w]),new_range))
        offset_w = min(2,offset_w-2)
        offset_h = min(2,offset_h-2)
        w = int(round(w*1.25))
        print(h,w)
    
    # for img in image_parts_values:
    #     cv2.imshow('a', img)
    #     cv2.waitKey(0)
    # print("Resizing...")
    # image_parts_values = [cv2.resize(img,(24,24)) for img in image_parts_values]
    # print("Normalizing...")
    # image_parts_values_normalized = list(map(varianceNormalize,image_parts_values))
    # print("Integratign...")
    # ii_parts_values =   
    print("Processing...")
    ii_parts_values =[cv2.integral(varianceNormalize(cv2.resize(img,(24,24))))[1:,1:] for img in image_parts_values]
    print("Detectiong...")
    all_detected_squares = [(image_parts_ranges[i],image_parts_values[i].shape) for i in Evaluator.predict(ii_parts_values)]
    print("Done...")
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

def tryOnImage(img_file,Evaluator,w):
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),2)
    cv2.imshow('resized',gray) 
    cv2.waitKey(0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    recs = detect(gray,Evaluator,w,2,2)
    recs = np.array([[recs[i][0][1],recs[i][0][0],recs[i][0][1]+recs[i][1][1],recs[i][0][0]+recs[i][1][0]] for i in range(len(recs))])
    recs = non_max_suppression_fast(recs,.1)
    [cv2.rectangle(img,(rec[0],rec[1]),(rec[2],rec[3]), (0, 0,255), 2) for rec in recs ]
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.imwrite('detected'+img_file,img)
    cv2.waitKey(0)


def main():
    Evaluator = Cascade('../Cascade/')
    #tryOnImage('faces.jpg',Evaluator,20)
    #cap = cv2.VideoCapture(0)
    while(True):
        cap = cv2.VideoCapture(0)
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame',gray)
        #cv2.waitKey(0);
        gray = cv2.blur(gray,(5,5))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        recs = detect(gray,Evaluator)
        recs = np.array([[recs[i][0][1],recs[i][0][0],recs[i][0][1]+recs[i][1][1],recs[i][0][0]+recs[i][1][0]] for i in range(len(recs))])
        recs = non_max_suppression_fast(recs,.1)
        [cv2.rectangle(frame,(rec[0],rec[1]),(rec[2],rec[3]), (0, 0,255), 2) for rec in recs ]
        
        cv2.imshow('frame',frame)
        if(cv2.waitKey(0) & 0xFF == ord('q')):
            break   
        cap.release()     
                        

    
    cv2.destroyAllWindows()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
 
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

