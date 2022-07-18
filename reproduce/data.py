#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import sys
sys.path.append('..')
from utils import util
import trip_seg


# In[1]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python data.ipynb   ')
    except:
        pass


# In[18]:


def _extract_hand_feat(segment):
    '''
    Extract 5 hand-crafted features as in 沈云. (2017). 
    基于 GPS 轨迹数据的交通出行方式识别研究 (Master's thesis, 北京交通大学). 
    Args:
        segment : [N,8]
    Returns:
        feat : [5,]
    '''
    feat = np.zeros((5,))
    feat[:3] = np.percentile(segment[:,1],[50,75,95])
    feat[3] = np.var(segment[:,1])
    feat[4] = np.sum(segment[:,0])
    return feat

def _extract_hand_feat_v1(segment):
    '''
    Extract first 11 hand-crafted features based only on GPS data.
    Ref: 
        Li, J., Pei, X., Wang, X., Yao, D., Zhang, Y. and Yue, Y., 2021. Transportation mode identification 
        with GPS trajectory data and GIS information. Tsinghua Science and Technology, 26(4), pp.403-416.
    Args:
        segment : [N,8]
    Returns:
        feat : [11,]
    '''
    feat = np.zeros((11,))
    # speed
    feat[0] = np.sum(segment[:,0])/np.sum(segment[:,7])
    feat[1] = np.var(segment[:,1])
    feat[2] = np.percentile(segment[:,1],85)
    # acceleration
    feat[3] = np.mean(segment[:,2])
    feat[4] = np.var(segment[:,2])
    feat[5] = np.percentile(segment[:,2],85)
    # head direction change
    hc = segment[1:,4] / segment[1:,7]
    feat[6] = np.mean(hc)
    feat[7] = np.var(hc)
    feat[8] = np.percentile(hc,85)
    # trajectory global features: total distance and ratio of low-speed points
    feat[9] = np.sum(segment[:,0])
    feat[10] = np.sum(segment[1:,1] <= 0.6)/(len(segment)-1)
    return feat

def _extract_deep_feat(segment,label,maxlen=200,minlen=10,num_modes=5):
    N,C = segment.shape
    num =  N//maxlen
    num2 = num+1 if N>=num*maxlen+minlen else num
    X = np.zeros((num2,maxlen,C))
    Y = np.zeros((num2,num_modes))
    for i in range(num):
        inx = i*maxlen
        X[i,:,:] = segment[inx:(inx+maxlen),:]
        Y[i,:] = _extract_seg_label(label[inx:(inx+maxlen)],num_modes)
    if num2 > num:  #  last part
        X[num,:(N-num*maxlen),:] = segment[(num*maxlen):,:]
        Y[num,:] = _extract_seg_label(label[(num*maxlen):],num_modes)
    return X,Y

def _extract_seg_label(label,num_modes):
    seg_label = np.zeros((num_modes,),dtype=np.int)
    for i in range(num_modes):
        seg_label[i] = np.sum(label == i)
    return seg_label

def _process_trip(trip,label,maxlen=None,minlen=10,seg='walk',feats_kind='hand',feats_index=[0,1,2,7]):
    '''
    Args:
        trip : (N,C),i.e., the segment of N GPS points with C pointwise features;
        maxlen : accepted maximal length of a segment after trip segmentation;
        minlen : accepted minimal length of a segment after trip segmentation;
        seg : method to segment a trip into segments,one of `walk`,`uniform`,`ws`, `label`,`pelt`;
        feats_kind : method to extract features, including hand-crafted and automatic features;
        feats_index: we need to use 4 pointwise features in order: relative distance, speed, 
            acceleration and time interval. 
    Returns:
        X : (N',num_feats) if `feats_kind` belongs to `hand`,else (N',maxlen,C);
        Y : (N',num_modes)
    '''
    masking = label >= 0
    trip,label = trip[masking],label[masking]
    tmp = trip[:,feats_index]
    #print(tmp.shape)
    #  received trip should be [(dist,v,a,t),...]
    cp_pos = trip_seg.divide_trip(tmp,label,seg,minlen)
    N,C = trip.shape
    num = len(cp_pos) + 1
    pos = [0]
    pos.extend(cp_pos)
    pos.append(N)
    num_modes = 5
    if feats_kind=='hand':
        num_feats = 5
        X = np.zeros((num,num_feats))
        Y = np.zeros((num,num_modes),dtype=np.int)  # [(label,cnt),...]
        for i in range(num):
            #inx = pos[i]:pos[i+1]
            X[i,:] = _extract_hand_feat(trip[pos[i]:pos[i+1],:])
            Y[i,:] = _extract_seg_label(label[pos[i]:pos[i+1]],num_modes)
    elif feats_kind=='hand_v1':
        num_feats = 11
        X = np.zeros((num,num_feats))
        Y = np.zeros((num,num_modes),dtype=np.int)
        for i in range(num):
            X[i,:] = _extract_hand_feat_v1(trip[pos[i]:pos[i+1],:])
            Y[i,:] = _extract_seg_label(label[pos[i]:pos[i+1]],num_modes)
#             Y[i,0] = label[pos[i]]
#             Y[i,1] = pos[i+1] - pos[i]
    elif feats_kind=='deep':
        X,Y = None,None
        for i in range(num):
            x,y = _extract_deep_feat(trip[pos[i]:pos[i+1],:],label[pos[i]:pos[i+1]],maxlen,minlen,num_modes)
            if X is None:
                X,Y = x,y
            else:
                X = np.concatenate((X,x),axis=0)
                Y = np.concatenate((Y,y),axis=0)
    else:
        raise ValueError('Unexpected feats_kind:',feats_kind)
    return X,Y

def _process_trips(X_ori,Y_ori,maxlen,minlen,seg='walk',feats_kind='hand'):
    X,Y = None,None
    N = len(X_ori)
    for i in range(N):
        x,y = _process_trip(X_ori[i,:,:],Y_ori[i,:],maxlen,minlen,seg,feats_kind)
        if X is None:
            X = x
            Y = y
        else:
            X = np.concatenate((X,x),axis=0)
            Y = np.concatenate((Y,y),axis=0)
    return X,Y


# In[7]:


def load_cls_data(path,maxlen=None,minlen=10,seg='walk',feats_kind='hand',only_test=False,seg_train='label'):
    '''
    Load trajectory data and divide trips into segments according to various method.
    Args:
        path : data path;
        maxlen : pre-defined maxlen of a segment;
        seg : segmentation method to divide trip, including `label`, `walk`, `ws`, `uniform`,`pelt`;
        seg_train : method to divide training trips
        feats_kind : type of features to be extracted.
    Returns:
        X_train :
        X_test :
        Y_train :
        Y_test : 
    Notes:
        The unput data should be trips with 8 channels and of length no greater than 2300 which are obtained
        after pre-processing. 8 channels are [`distanc`,`velocity`,`acceleration`,`jerk`,`bearing rate`,
        `delta latitude`,`delta longitude`,`delta time`] in order. 
    '''
    X_train_ori,X_test_ori,Y_train_ori,Y_test_ori = util.load_pickle(path)
    # dealing test set
    X_test,Y_test = _process_trips(X_test_ori,Y_test_ori,maxlen,minlen,seg,feats_kind)
    if only_test:
        return X_test,Y_test
    # dealing training set
    X_train,Y_train = _process_trips(X_train_ori,Y_train_ori,maxlen,minlen,seg_train,feats_kind)
    return X_train,X_test,Y_train,Y_test


# In[4]:


def count_two_mode_seg(Y):
    # Y : [N,5]
    num = len(Y)
    cnt = 0
    for i in range(num):
        n = np.sum(Y[i,:]>0)
        cnt += 1 if n > 1 else 0
    print('total %d/%d (%.4f) segments have more than one modes'%(cnt,num,cnt/num))


# In[16]:


if __name__ == '__main__':
    # data of the path have 8 features : [(dist,v,a,jerk,bearingRate,delta_lat,delta_lng,delta_t),...]
    '''
    After data pre-processing, each trip is of length less than 2300 and 8 channels, obtained from `DL_data_creation.py`.
    '''
    path2 = '../../TransModeDetection/data/feats_maxlen_2300_8F_wos.pickle'
    feats_index = [0,1,2,7] 
    seg_list = ['uniform','ws','walk']
    for i in range(len(seg_list)):
        X_test,Y_test = load_cls_data(path2,None,10,seg_list[i],'hand',only_test=True)
        print(seg_list[i],X_test.shape,Y_test.shape)
        count_two_mode_seg(Y_test)
        




