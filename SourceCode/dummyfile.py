
import numpy as np
"""
def setOptimalThreshold(scores,num_pos):
    num_neg=len(scores)-num_pos
    num_pos_before=0;num_neg_before=0;err = -1;err_indx=-1
    pos_neg_flags = np.array(range(0,len(scores))) < num_pos
    scores_with_flags = sorted(zip(scores,pos_neg_flags),key=lambda x: x[0])
    print(scores_with_flags)
    for i in range(len(scores_with_flags)):
        if(scores_with_flags[i][1]): 
            num_pos_before+=1 
        else:
            num_neg_before+=1
        new_err = min(num_pos_before+num_neg - num_neg_before ,num_neg_before + num_pos-num_pos_before)
        print(num_pos_before+num_neg - num_neg_before)
        if(err_indx ==-1 or new_err < err ):
            err=new_err
            err_indx=i

    if(err_indx < len(scores)-1):
        return scores_with_flags[err_indx][0]+(scores_with_flags[err_indx+1][0] - scores_with_flags[err_indx][0])/2.
    return scores_with_flags[err_indx]



print(setOptimalThreshold([3,1,20,35,9,8,19,7,25],5))

from multiprocessing import Pool
def add(p):
    x,y = p
    return x+y

if __name__ == '__main__':
    data_pairs = [ [3,5], [4,3], [7,3], [1,6] ]
    p = Pool(5)
    a =[1,2,3,4,5,6]
    b = a.copy()
    result = p.map(add,zip(a,b))
    print(result)
"""
import itertools   
w = 2
h = 2
offset_h = 1
offset_w = 1
ii = np.array(
    [[1,2,3,10],
    [4,5,6,11],
    [6,7,8,12],
    [61,72,84,132]])
while(w < ii.shape[0] and h<ii.shape[1]):
    r = list(range(0,ii.shape[0]+offset_h, h -1 + offset_h))
    r = [ (i,j-offset_h) for (i,j) in zip(r,r[1:])]
    c = list(range(0,ii.shape[1]+offset_w, w -1 + offset_w))
    c = [ (i,j-offset_w) for (i,j) in zip(c,c[1:])]
   
    
    ii_parts_ranges = [rectangle for rectangle in itertools.product(r,c) ]
    ii_parts_values = [ii[p[0][0]:p[0][1]+1,p[1][0]:p[1][1]+1] for p in ii_parts_ranges]
    print(ii_parts_values)
    w *= int(w*1.5)
    h *= int(h*1.5)
    offset_h += 1
    offset_w += 1
