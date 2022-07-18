#!/usr/bin/env python
# coding: utf-8

'''
This file create training, validation and test dataset for our model,
using data from another file `pre_processing` which caculate the trip's 
features. 
Inputs:
    `Feats Trip` from `pre_processing.py`.
Outputs:
    Trips of length between N_min and N_max. All trips are padded with zero to be
        same length of N_max. Data are splited into training, validation and test set
        but saved in a pickle file.
'''

import os
import time
import numpy as np
import random
import pickle
from utils import util

#   1. Basic funcitons

##  1.1 Get index of training, validation and test set.
def get_train_test_index(y,train_ratio=0.7,val_ratio=0.1,stratify=None):
    '''
    Attain index of train,validation,test data set.
    :param y: int,1-d array or list,labels of spliting data;
    :param train_ratio: ratio of training data;
    :param val_ratio:ratio of validation data;
    :param stratify : whether keep ratio of each label,default is None.
        if None,don't keep ratio else otherwise;
    :return index_train,index_val,index_test:list,index of training,validation,
        test data set.
    '''
    if stratify is None:
        N = len(y)
        index = list(range(N))
        random.shuffle(index)
        return index[0:round(N*train_ratio)],\
               index[round(N*train_ratio):round(N*(train_ratio+val_ratio))],\
               index[round(N*(train_ratio+val_ratio)):]
    # need to keep ratio of each label.
    arr = np.array(y,dtype=np.int32)
    index_train = []
    index_val = []
    index_test = []
    for k in np.unique(arr):
        index = np.argwhere(arr == k)
        index = index.reshape(index.shape[0])
        random.shuffle(index)
        N = len(index)
        index_train.extend(list(index[0:round(N*train_ratio)]))
        index_val.extend(list(index[round(N*train_ratio):round(N*(train_ratio+val_ratio))]))
        index_test.extend(list(index[round(N*(train_ratio+val_ratio)):]))
    random.shuffle(index_train)
    random.shuffle(index_val)
    random.shuffle(index_test)
    return index_train,index_val,index_test

##  1.2 truncate the long trip and pad the short trip so that all prosessed trips
#       have same length (timesteps). The labels of padded points are set to -1.
def truncate_pad_seq(seq,maxlen,padding='post',value=0,minlen=20,label=None):
    '''
    `seq` must be a list or iterables.Here,seq is a 1-d list.
    :out [valid_samples,maxlen] 
    '''
    def pad_seq(seq_old,seq_new):
        if padding == 'post':
            if length_new >= length:
                seq_new[0:length] = seq_old
            else:
                seq_new[0:length_new] = seq_old[0:length_new]
        elif padding == 'pre':
            if length_new >= length:
                seq_new[(length_new-length):] = seq_old
            else:
                seq_new[(length-length_new):] = seq_old[0:length_new]
        else:
            raise ValueError('Unexpected padding method:',padding)
        seq_new = seq_new.reshape(n_samples,maxlen)
        return seq_new
        
    length = len(seq)
    quotient = length//maxlen
    remainder = length - quotient*maxlen
    n_samples = quotient if remainder < minlen else quotient+1
    length_new = n_samples * maxlen
    arr = np.ones((length_new,)) * value
    arr = pad_seq(seq,arr)
    len_list = [maxlen for _ in range(quotient)]
    if remainder >= minlen:
        len_list.append(remainder)
    if label is None:
        return arr,len_list
    # truncate and pad label
    label_new = - np.ones((length_new,),dtype=np.int32)
    label_new = pad_seq(label,label_new)
    return arr,len_list,label_new


##  1.3 five-spot triple smoothing method
def smooth5_3(y):
    '''
    Smoothing time series with five point three order smoothing method.
    :param y : array-like,1-d time series;
    :return : smoothed time series .
    '''
    y_in = np.array(y)
    N = len(y_in)
    if N < 5:
        return y_in
    y_out = np.zeros(y_in.shape)
    y_out[0] = (69*y_in[0] + 4*y_in[1] - 6*y_in[2] + 4*y_in[3] - y_in[4])/70
    y_out[1] = (2*y_in[0]  + 27*y_in[1]+12*y_in[2] - 8*y_in[3] + 2*y_in[4])/30
    for i in range(2,N-2):
        y_out[i] = (-3*y_in[i-2] + 12*y_in[i-1] + 17*y_in[i] + 12*y_in[i+1] - 3*y_in[i+2])/35
    y_out[N-2] = (2*y_in[N-5] - 8*y_in[N-4] + 12*y_in[N-3] + 27*y_in[N-2] + 2*y_in[N-1])/35
    y_out[N-1] = (-y_in[N-5] + 4*y_in[N-4] -6*y_in[N-3] + 4*y_in[N-2] + 69*y_in[N-1])/70
    return y_out





#   2. Split data into training and test set

def fixed_len_trip(data_dir,output_path,maxlen,features=['speed'],minlen=20,train_ratio=0.7,val_ratio=0.1):
    '''
    fixed length of trip with truncate and pad.
    Resulted data has shape of [n_samples,timesteps,n_features], label has shape
    of [n_samples,timesteps].
    '''
    start_time = time.process_time()
    feats_dict = {'distance':0,
                'speed':1,
                'acc':2,
                'jerk':3,
                'delta_lat':4,
                'delta_lng':5,
                'delta_time':6,
                'bearing_rate':7,
                'label':8}
    index_smooth = [0,1,2,3,7]
    column_index = [feats_dict[feat] for feat in features]
    users_list = os.listdir(data_dir)
    feats_all_user = None
    label_all_user = None
    for user in users_list:
        user_data_dir = data_dir + user + '/'
        file_list = os.listdir(user_data_dir)
        for file in file_list:
            feats_one_trip = util.load_csv2list(user_data_dir + file,is_traj=False)
            feats_one_trip = np.array(feats_one_trip)
            label = feats_one_trip[:,feats_dict['label']]
            label = label.astype(np.int32)
            feats = None
            for index in column_index:
                # smooth --> trauncate/pad --> expand dim
                #[n_pts,feats] --> [samples,maxlen,feats]
                feat = feats_one_trip[:,index] # [n_pts,]
                if index in index_smooth:
                    feat = smooth5_3(feat)
                if feats is None:
                    feat,_,label = truncate_pad_seq(feat,maxlen=maxlen,minlen=minlen,label=label)
                    feat = feat[:,:,np.newaxis]
                    feats = feat
                else:
                    feat,_ = truncate_pad_seq(feat,maxlen=maxlen,minlen=minlen)
                    feat = feat[:,:,np.newaxis]
                    feats = np.concatenate((feats,feat),axis=2)            
            if feats_all_user is None:
                feats_all_user = feats # [samples,maxlen,feats]
                label_all_user = label # [samples,maxlen]
            else:
                feats_all_user = np.concatenate((feats_all_user,feats),axis=0)
                label_all_user = np.concatenate((label_all_user,label),axis=0)
    # split data into train,validation,test set
    index_train,index_val,index_test = get_train_test_index(label_all_user,train_ratio=train_ratio,val_ratio=val_ratio)
    X , Y = feats_all_user,label_all_user
    print('X and Y:',X.shape,Y.shape)
    if index_val == []:
        print('Only train and test:',len(index_train),len(index_test))
        with open(output_path,'wb') as f:
            pickle.dump([X[index_train],X[index_test],Y[index_train],Y[index_test]],f)
    else:
        with open(output_path,'wb') as f:
            pickle.dump([X[index_train],X[index_val],X[index_test],
                         Y[index_train],Y[index_val],Y[index_test]],f)
    print('consuming time:',time.process_time()-start_time,'second.')


if __name__ == '__main__':
    """ 
    Because original trip lengths range from 20 to 40000+, which is harmful 
    for spliting training and test set. To takcle this, we divide long trips 
    into non-overlapped trips of length no greater than 2300 (i.e., 95th percentile 
    of original lengths distribution). 
    Then, these length-limited trips are splited into training and test set.
    For two-stage methos, various trip segmentation (change points detection) 
    methods are applied to divide trip into single-one mode segments. Then, classify
    these segments.
    For, one-stage method or ours(TrajYOLO, TrajSSD), the trip length needs limited 
    further described following.
    """
    fixed_len_trip(data_dir = '../data/Feats Trip/',
                   output_path = '../data/trips_fixed_len_2300_8F.pickle',
                   maxlen = 2300,
                   minlen = 20,
                   features = ['distance','speed','acc','jerk','bearing_rate','delta_lat','delta_lng','delta_time'])
                   
                   

##  3. Limit trip length between N_min and N_max for TrajYOLO and TrajSSD
#      Training set are further splited into training and validation set so that 
#       training:validation:test = 7:1:2.
def limit_trip_len(X,Y,maxlen,minlen=20):
    masking = Y >=0
    C = X.shape[2]
    size_padding = int(np.ceil(X.shape[1]/maxlen)*maxlen - X.shape[1])
    if size_padding > 0:   
        tmp = np.zeros((X.shape[0],size_padding,X.shape[2]),dtype=np.float)
        X = np.concatenate((X,tmp),axis=1)
        X = X.reshape(-1,maxlen,C)
        tmp = - np.ones((Y.shape[0],size_padding),dtype=np.int)
        Y = np.concatenate((Y,tmp),axis=1)
        Y = Y.reshape(-1,maxlen)
        tmp = np.zeros((masking.shape[0],size_padding),dtype=bool)
        masking = np.concatenate((masking,tmp),axis=1)
        masking = masking.reshape(-1,maxlen)
    else:
        X = X.reshape(-1,maxlen,C)
        Y = Y.reshape(-1,maxlen)
        masking = masking.reshape(-1,maxlen)
    index_del = []
    index_drop = []
    for i in range(len(masking)):
        num_valid = np.sum(masking[i])
        if num_valid < minlen:
            index_del.append(i)
            if num_valid > 0:
                index_drop.append(i)
    X = np.delete(X,index_del,axis=0)
    Y = np.delete(Y,index_del,axis=0)
    return X,Y,index_drop

def get_fixed_len_trip(output_path,data_path,maxlen=400,feats_index=None):
    """
    X : [n_samples,maxlen,8]
    Y : [n_samples,maxlen], one-hot
    """
    with open(data_path,'rb') as f:
        X_train,X_test,Y_train,Y_test = pickle.load(f)
    X_train,Y_train,index_drop = limit_trip_len(X_train,Y_train,maxlen=maxlen)
    print('in training set,drop:',len(index_drop),' samples:',X_train.shape)
    X_test,Y_test,index_drop = limit_trip_len(X_test,Y_test,maxlen=maxlen)
    print('in test set,drop:',len(index_drop),' samples:',X_test.shape)
    if feats_index is not None:
        X_train = X_train[:,:,feats_index]
        X_test = X_test[:,:,feats_index]
    # spilit training set into training set and test set
    ratio = 0.875
    N = X_train.shape[0]
    index = list(range(N))
    random.shuffle(index)
    index_train = index[0:round(N*ratio)]
    index_val = index[round(N*ratio):]
    with open(output_path,'wb') as f:
        pickle.dump([X_train[index_train],X_train[index_val],X_test,
                     Y_train[index_train],Y_train[index_val],Y_test],f)
                     
if __name__ == '__main__':
    data_path = '../data/feats_maxlen_2300.pickle'
    get_fixed_len_trip(output_path = '../data/trips_fixed_len_400_8F.pickle',
                       data_path = data_path,
                       feats_index = None,
                       maxlen = 400)