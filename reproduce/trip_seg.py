#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import scipy
from matplotlib import pyplot as plt
import ruptures as rpt

import sys
sys.path.append('..')
from utils import util


# In[17]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python trip_seg.ipynb   ')
    except:
        pass


# # 1. Divide trip into segments

# ## 1.0 interface 

# In[12]:


def divide_trip(trip,label=None,seg='label',thd_len=20):
    '''
    Receive trip and return positions of change points
    Args:
        trip : point feat of each point,[(dist,v,a,t)];
        label : trans mode of the trip;
        seg : method to divide the trip;
    Return:
        cp_pos : position of change points.
    '''
    if seg == 'label':
        cp_pos = find_change_points(label)
    elif seg == 'uniform':
        cp_pos = divide_trip_uniform(trip)
    elif seg == 'walk':
        label_t = divide_trip_walk(trip)
        cp_pos = find_change_points(label_t)
    elif seg == 'ws':
        cp_pos = divide_trip_ws(trip)
    elif seg == 'pelt':
        cp_pos = divide_trip_pelt(trip)
    else:
        raise ValueError('Unexpected seg method:',seg)
    return correct_cp_pos(cp_pos,len(trip),thd_len)


# In[13]:


def correct_cp_pos(cp_pos,N,thd_len=20):
    '''Make sure each segment contains at least `thd_len` GPS points.'''
    cp_pos_c = []
    num = len(cp_pos)
    if thd_len <= 0 or num < 1:
        return cp_pos
    if cp_pos[0] >= thd_len and N-cp_pos[0] >= thd_len:
        cp_pos_c.append(cp_pos[0])
    inx = 0
    for i in range(num-1):
        if cp_pos[i]-cp_pos[inx]>=thd_len:
            cp_pos_c.append(cp_pos[i])
            inx = i
    if N-cp_pos[num-1] >= thd_len and cp_pos[num-1]-cp_pos[inx]>=thd_len:
        cp_pos_c.append(cp_pos[num-1])
    return cp_pos_c


# ## 1.1 uniform segmentation

# In[33]:


def divide_trip_uniform(trip,T=120):
    """ Divide the trip into segments using uniform time window. The best
        time window size is 120s (Zheng et al., 2010, see `divide_trip_walk`)"""
    cp_pos = []
    i = 0
    N = len(trip)
    while i < N:
        j = i + 1
        while j <= N:
            duration = np.sum(trip[i:j,3])
            if duration > T:
                break
            j += 1
        # a segment contains at least 3 points
        j = i+3 if j<i+3 else j
        if j > N:
            break
        cp_pos.append(j)
        i = j
    return cp_pos

def divide_trip_uniform_v1(trip,T=20):
    cp_pos = []
    N = len(trip)
    num = N // T
    for i in range(num):
        cp_pos.append((i+1)*T)
    return cp_pos


# ## 1.2 walk segment based segmentation 

# In[34]:


def divide_trip_walk(trip,v_max=2.5,a_max=1.5,thd_dist1=30,thd_t=10,thd_dist2=200,thd_us=3):
    '''
    Desc:
        Implementation of walk-segment based trip segmentation method proposed in: 
        Yu Zheng, Yukun Chen, Quannan Li, Xing Xie, and Wei-Ying Ma. 2010. Understanding 
        transportation modes based on GPS data for web applications. <i>ACM Trans. Web</i> 
        4, 1, Article 1 (January 2010), 36 pages. DOI:https://doi.org/10.1145/1658373.1658374
    Args:
        trip : (N,4), attrs array of a trip,[[dist,v,a,t],...];
        v_max : max velocity of walk points;
        a_max : max acceleration of walk points;
        thd_dist1 : segment with dist less than `thd_dist1` is merged into its 
            backward segment;
        thd_t : segment with duration less than `thd_t` is merged into its
            backward segment;
        thd_dist2 : distance threshold for distinguishing certain segments (cs)
            from uncertain segments (us);
        thd_us : if number of consecutive us exceeds `thd_us`, these us will be
            merged into a non-walk segment.
    Returns:
        label : 1-d array with same length of trip, denote whether a point belongs to 
            walk or non-walk.
        
    '''
    N,_ = trip.shape
    label = np.ones((N,),dtype=np.int) # 0 for walk point, 1 for non-walk
    # step 1 : distinguish walk-point from non-walk point
    for i in range(N):
        if trip[i][1] < v_max and np.abs(trip[i][2]) < a_max:
            label[i] = 0
    # step 2 : merge segment whose dist less than `thd_dist1`  into its backward segment
    last_label,last_len,cur_len = label[0],0,1
    for i in range(1,N):
        if label[i] == label[i-1]:
            cur_len += 1
            if i < N-1:
                continue
        if last_len > 0:  # start from second segment
            cur_dist = np.sum(trip[(i-cur_len):i,0])
            cur_duration = np.sum(trip[(i-cur_len):i,3])
            if cur_dist < thd_dist1 or cur_duration < thd_t:
                label[(i-cur_len):i] = last_label
        last_len = 1
        cur_len = 1
        last_label = label[i-1]
    # step 3 : merge uncertain segments
    num_us,inx_srt = 0,0
    cnt = 0
    for i in range(1,N):
        if label[i] == label[i-1]:
            cnt += 1
            if i < N-1:
                continue
        dist = np.sum(trip[(i-cnt):i])
        if dist >= thd_dist2:
            if num_us >= thd_us:
                label[inx_srt:i] = 1
                inx_srt = i
            num_us = 0
        else:
            num_us += 1
            if i == N-1 and num_us > thd_us:  # the last segment
                label[inx_srt:i] = 1
    return label


# ## 1.3 window and similarity based segmentation 

# In[42]:


def divide_trip_ws(trip,T=90,eta=0.05,delta=1):
    '''
    Find change points besed on windows and similarity method.
    Ref : Shen et al. (2015, September). A method of traffic travel status segmentation 
    based on position trajectories. In 2015 IEEE 18th International Conference on 
    Intelligent Transportation Systems (pp. 2877-2882). IEEE.
    Args:
        trip : [(dist,v,a,t),...];
        T : time window size;
        eta : coeff deciding threshold of max similarity;
        delta : minimum difference of serie number;
    Returns:
        cp_pos : list of positions of change points.
    '''
    srt_list,sim_list = split_trip_and_calc_sim(trip,T)
    if len(srt_list)>0:
        srt_list.pop(0)
    if len(sim_list) < 2:
        return srt_list
    flag = np.zeros((len(sim_list),)) # 1 for potential change point
    # step1 : define initial change point
    inx = scipy.signal.find_peaks(sim_list,distance=delta)
    inx = inx[0]
    N = len(inx)
    if N == 0:
        return []
    elif N == 1:
        return [srt_list[inx[0]]]
    else:
        cp_pos = []
        dmax = np.max(sim_list[inx])
        for i in range(N):
            if sim_list[inx[i]] >= eta*dmax:
                flag[inx[i]] = 1
    # step2 : correct change points
    inx1,inx2 = -1,-1
#     for i in range(len(flag)):
#         if flag[i] <= 0:
#             continue
#         if inx1 == -1:
#             inx1 = i
#             continue
#         inx2 = i
#         if inx2 - inx1 < delta:
#             flag[inx2] = 0
#         else:
#             inx1 = inx2
        
    for i in range(len(flag)):
        if flag[i] > 0:
            cp_pos.append(srt_list[i])
    return cp_pos
    
def split_trip_and_calc_sim(trip,T):
    min_len = 5  # minimum number of GPS points in a segment
    srt_list = []
    i = 0
    N = len(trip)
    feats = []
    while i < N:
        j = i + 1
        while j <= N:
            duration = np.sum(trip[i:j,3])
            if duration > T:
                break
            j += 1
        # a segment contains at least 3 points
        j = i+min_len if j<i+min_len else j
        if j > N:
            break
        # extract features
        feat = np.zeros((5,))
        feat[:3] = np.percentile(trip[i:j,1],[50,75,95])
        feat[3] = np.var(trip[i:j,1])
        #feat[4] = np.sum(trip[i:j,0])
        feats.append(feat)
        # update 
        srt_list.append(i)
        i = j
    # calculate similarity
    sim_list = []
    N = len(srt_list)
    if N > 1:
        for i in range(1,N):
            sim = np.sqrt(np.sum(np.power(feats[i-1]-feats[i],2)))
            #sim = np.exp(-sim)
            sim_list.append(sim)
    return srt_list,np.array(sim_list)


# ## 1.4 PELT
# pruned extract linear time

# In[27]:


def divide_trip_pelt(trip,pen=16,maxlen=248):
    '''
    Find change points besed on PELT method.
    Ref : Dabiri, Sina, et al. "Semi-Supervised Deep Learning Approach for Transportation
    Mode Identification Using GPS Trajectory Data." IEEE Transactions on Knowledge and Data Engineering (2019).
    Args:
        trip : [(dist,v,a,t),...];
        pen : penalty. 
    Returns:
        cp_pos : list of positions of change points.
    '''
    signal = trip[:,1:3]  # only velocity and acceleration are used
    minlen = 10
    N = len(signal)
    cp_pos = []
    num = N // maxlen
    num = num+1 if N>num*maxlen+minlen else num
    start = 0
    for i in range(num):
        end = N if i == num-1 else start+maxlen
        algo = rpt.Pelt(model="l2").fit(signal[start:end])
        res = algo.predict(pen=pen)  # the last one is length of the signal
        for j in range(0,len(res)-1):
            cp_pos.append(res[j]+start)
        start += maxlen
    return cp_pos


# In[28]:


res = divide_trip_pelt(signal)
print(res)


# # 2. Evaluate segmentation method 

# ## 2.1 basic evaluate function

# In[29]:


def find_change_points(label):
    '''
    Find all change points of a trip. Change point is the point whose label
    if different from its previous one. Point index starts from 0.
    label : 1-d array.
    '''
    cp_pos = []
    for i in range(1,len(label)):
        if label[i] == -1:
            break
        if label[i] != label[i-1]:
            cp_pos.append(i)
    return cp_pos

def calc_TP(cp_pr,trip,label,thd_dist=150,seg='walk'):
    '''
    If predicted change point and true cp are within `thd_dist`,
    the pr is thought as accurate.
    '''
    masking = label >= 0
    cp_gt = find_change_points(label)
    flag = [0 for _ in range(len(cp_pr))]
    tp_cnt = 0
    for i in range(len(cp_gt)):
        if len(cp_pr) == 0:
            break
        dist_list = []
        for j in range(len(cp_pr)):
            inx_srt = min(cp_gt[i],cp_pr[j])
            inx_end = max(cp_gt[i],cp_pr[j])
            if inx_srt == inx_end:
                dist = 0
            else:
                dist = np.sum(trip[inx_srt:inx_end,0])
            dist_list.append(dist)
        inx = np.argmin(dist_list)
        if flag[inx]==0 and dist_list[inx]<thd_dist:
            tp_cnt += 1
            flag[inx] = 1
    return tp_cnt,len(cp_pr),len(cp_gt)

def calc_seg_pr_re(trips,labels,thd_dist=150,seg='walk',thd_len=0,outputs=None):
    # `outputs` is prediction of labels 
    N = len(trips)
    cnt = np.zeros((3,))
    for i in range(N):
        masking = labels[i]>=0
        if outputs is None:
            cp_pr = divide_trip(trips[i][masking],labels[i],seg,thd_len)
        else:
            cp_pr = divide_trip(trips[i][masking],outputs[i],'label',thd_len)
        tp_cnt,num_pr,num_gt = calc_TP(cp_pr,trips[i],labels[i],thd_dist,seg)
        cnt[0] += tp_cnt
        cnt[1] += num_pr
        cnt[2] += num_gt
    recall = cnt[0] / cnt[2]
    precision = cnt[0] / cnt[1]
    print('total',N,'trips,recall and presicion of change points are:',recall,precision)
    print('TP,TP+FP,TP+FN:',cnt)


# In[36]:


def calc_seg_len(trips,labels,seg='walk',thd_len=10,bins=40):
    N = len(trips)
    seg_len_list = []
    maxlen = 600
    for i in range(N):
        masking = labels[i] >= 0
        cp_pos = divide_trip(trips[i][masking],labels[i][masking],seg,thd_len)
        if len(cp_pos)==0:
            seg_len_list.append(min(np.sum(masking),maxlen))
            continue
        cp_pos.append(np.sum(masking))
        cp_pos.insert(0,0)
        for j in range(1,len(cp_pos)):
            seg_len_list.append(min(cp_pos[j] - cp_pos[j-1],maxlen))
    # plot histgram
    print('minimum, maximum, mean of segment length:',np.min(seg_len_list),np.max(seg_len_list),np.mean(seg_len_list))
    plt.hist(seg_len_list,bins=bins,density=False,facecolor='blue',edgecolor='black')
    plt.xlabel('seg length')
    plt.ylabel('fre')
    plt.title('seg length distribution using %s'%(seg))
    plt.show()
    
def pred2label(path):
    '''
    pred : [N,]
    masking : [B,L]
    '''
    pred,masking = util.load_pickle(path)
    masking = masking.numpy()
    labels = - np.ones((masking.shape[0],masking.shape[1]),dtype=np.int)
    cnt = 0
    for i in range(masking.shape[0]):
        num = np.sum(masking[i])
        labels[i][:num] = pred[cnt:(cnt+num)]
        cnt += num
    return labels


# ## 2.2 evaluation instance 

if __name__ == '__main__':
    path = '../../TransModeDetection/data/feats_maxlen_2300_8F.pickle'
    _,X_test,_,Y_test = util.load_pickle(path)
    feats_index = [0,1,2,7] 
    thd_len = 20
    calc_seg_pr_re(X_test[:,:,feats_index],Y_test,150,'uniform',thd_len)
    calc_seg_pr_re(X_test[:,:,feats_index],Y_test,150,'walk',thd_len)
    calc_seg_pr_re(X_test[:,:,feats_index],Y_test,150,'ws',thd_len)
    calc_seg_pr_re(X_test[:,:,feats_index],Y_test,150,'pelt',thd_len)


# In[45]:


if __name__ == '__main__':
    calc_seg_len(X_test[:,:,feats_index],Y_test,'label',10,bins=40)
    calc_seg_len(X_test[:,:,feats_index],Y_test,'uniform',10,bins=40)
    calc_seg_len(X_test[:,:,feats_index],Y_test,'walk',10,bins=40)
    calc_seg_len(X_test[:,:,feats_index],Y_test,'ws',10,bins=40)
    calc_seg_len(X_test[:,:,feats_index],Y_test,'pelt',10,bins=40)


# In[36] :




