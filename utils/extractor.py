#!/usr/bin/env python
# coding: utf-8

# In[16]:


import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn import preprocessing
import pywt


# In[18]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python extractor.ipynb   ')
    except:
        pass


# # 1. Hand-crafted features

# In[ ]:


class IdentityFeat(nn.Module):
    def __init__(self,last_dim):
        super(IdentityFeat,self).__init__()
        self.last_dim = last_dim
        
    def forward(self,x):
        # x : [B,C] 
        return x


# In[2]:


class IdentityScaler():
    def __init__(self):
        pass
    def transform(self,x):
        return x
    
def minmax_scaler(X):
    # X : [B,n_feats]
    x_min = np.min(X,axis=0)
    x_max = np.max(X,axis=0)
    den = np.power(x_max - x_min,-1)
    return np.multiply(X-x_min,den)

def std_scaler(X):
    # X : [B,n_feats]
    x_std = np.std(X,axis=0)
    x_mean = np.mean(X,axis=0)
    x_std = np.power(x_std,-1)
    return np.multiply(X-x_mean,x_std)


# In[3]:


def _extract_hand_feat_v1(segment):
    '''
    Extract first 11 hand crafted features based on only GPS data.
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

def extract_feats(X,Y):
    '''
    Extract 11 features for each sub-trip, i.e., mean, variation and 85th percentile
        of speed, acceleration and head direction change, total distance and ratio of low-speed 
        points.
    Args:
        X : [B,L,8];
        Y : [B,L]
    Returns:
        feats : [B,num_feats], 
    '''
    B,_ = Y.shape
    feats = np.zeros((B,11))
    percentiles = np.array([95,75,50])
    for i in range(B):
        masking = Y[i] >= 0
        feats[i] = _extract_hand_feat_v1(X[i][masking])
    return feats


def load_linreg_data(data_path,is_train=False,norm_type=None):
    with open(data_path,'rb') as f:
        X_train,X_val,X_test,Y_train,Y_val,Y_test = pickle.load(f)
    num_train,num_val,num_test = len(X_train),len(X_val),len(X_test)
    # feats : [B,num_feats]
#     print(X_train.shape,X_val.shape,X_test.shape,Y_train.shape,Y_val.shape,Y_test.shape)
    feats = extract_feats(np.concatenate((X_train,X_val,X_test),axis=0),
                          np.concatenate((Y_train,Y_val,Y_test),axis=0))
    if norm_type == 'norm':
        scaler = preprocessing.StandardScaler().fit(feats[:(num_train+num_val),:])
    elif norm_type == 'minmax':
        scaler = preprocessing.MinMaxScaler().fit(feats[:(num_train+num_val):,:])
    else:
        scaler = IdentityScaler()
    if is_train:
        X_train = scaler.transform(feats[:num_train,:])
        X_val = scaler.transform(feats[num_train:(num_train+num_val),:])
        return torch.from_numpy(X_train).float(),torch.from_numpy(X_val).float(),                torch.from_numpy(Y_train).long(),torch.from_numpy(Y_val).long()
    else:
        X_test = scaler.transform(feats[(num_train+num_val):,:])
        return torch.from_numpy(X_test).float(),torch.from_numpy(Y_test).long()


# In[8]:


if __name__ == "__main__":
    path = 'D:/Anaconda/documents/TransModeDetection/data/trips_fixed_len_400_8F.pickle'
    X_train,X_test,Y_train,Y_test = load_linreg_data(path,norm_type='norm')
    print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# # 2. Discrete Wavelet Transformation

# In[16]:


def DWT(x,wavelet='db4',F=8):
    # extract the maximum,minimum,mean,std of the approximation 
    # signal A_M(t)
    M = int(np.log(len(x)/(F-1))/np.log(2))
    coeffs = pywt.wavedec(x, wavelet, level=M)
    for i in range(1,M+1):
        coeffs[i][:] = 0.
    AM_t = pywt.waverec(coeffs, wavelet)
    return [np.max(AM_t),np.min(AM_t),np.mean(AM_t),np.std(AM_t)]

def extract_dwt(x,y,wavelets=['db4'],F=[8]):
    B,L,C = x.shape
    masking = y >= 0
    num_wave = len(F)
    feats = torch.zeros(B,C*num_wave*4)
    for i in range(B):
        tmp = []
        for k in range(C):
            x_tmp = x[i][masking[i]]
            for j in range(num_wave):
                tmp.extend(DWT(x_tmp[:,k],wavelets[j],F[j]))
        feats[i,:] = torch.Tensor(tmp)
    return feats

def extract_dwt_v1(x,y,wavelets=['db4'],F=[8]):
    # x : [B,L,C]
    # y : [N,k]
    B,L,C = x.shape
    lens = torch.sum(y,1)
    num_wave = len(F)
    feats = torch.zeros(B,C*num_wave*4)
    for i in range(B):
        tmp = []
        for k in range(C):
            x_tmp = x[i,:lens[i]]
            for j in range(num_wave):
                tmp.extend(DWT(x_tmp[:,k],wavelets[j],F[j]))
        feats[i,:] = torch.Tensor(tmp)
    return feats


# In[6]:


if __name__ == '__main__':
    x = torch.rand(10)
    out = DWT(x,'sym2',4)
    print(out)
    x = torch.rand(10,40,3)
    y = torch.ones(10,40)
    feats = extract_dwt(x,y)
    print(feats.shape)


# In[21]:


class DWTFeat(nn.Module):
    def __init__(self,
                 n_feats = 3,
                 device = "cpu",
                 y_fmt = None
                ):
        super(DWTFeat,self).__init__()
        self.wavelets = ['db1','db2','db3','db4','sym2','sym3','sym4']
        self.F = [2,4,6,8,4,6,8]
        self.last_dim = n_feats * len(self.F) * 4
        self.device = device
        self.y_fmt = y_fmt
    
    def forward(self,x,y):
        #  x : [B,L,C] --> [B,C_out]
        #  y : [B,L] if y_fmt is None else [B,k]
        if self.y_fmt is None:
            x = extract_dwt(x,y,self.wavelets,self.F)
        else:
            x = extract_dwt_v1(x,y,self.wavelets,self.F)
        return x.to(self.device)


# In[22]:


if __name__ == '__main__':
    x = torch.rand(10,40,4)
    y = torch.ones(10,40)
    cls = DWTFeat()
    feats = cls(x,y)
    print(feats.shape)
    y = (torch.ones(10,5)*30).long()
    cls = DWTFeat(y_fmt='v1')
    feats = cls(x,y)
    print(feats.shape)


# In[ ]:




