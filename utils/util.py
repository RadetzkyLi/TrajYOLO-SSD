#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch.autograd import Variable
import pickle
from sklearn import preprocessing


# In[1]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python util.ipynb   ')
    except:
        pass


# In[ ]:


def save_as_pickle(file_path,data):
    with open(file_path,'wb') as f:
        pickle.dump(data,f)

def load_pickle(file_path):
    with open(file_path,'rb') as f:
        return pickle.load(f)


# In[5]:


def load_seg_data(data_path,is_train=True):
    if is_train:
        with open(data_path,'rb') as f:
            X_train,X_val,_,Y_train,Y_val,_ = pickle.load(f)
        X_train = Variable(torch.from_numpy(X_train).float())
        X_val = Variable(torch.from_numpy(X_val).float())
        Y_train = Variable(torch.from_numpy(Y_train).type(torch.long))
        Y_val = Variable(torch.from_numpy(Y_val).type(torch.long))
        return X_train,X_val,Y_train,Y_val
    else:
        with open(data_path,'rb') as f:
            _,_,X_test,_,_,Y_test = pickle.load(f)
        X_test = Variable(torch.from_numpy(X_test).float())
        Y_test = Variable(torch.from_numpy(Y_test).float())
        return X_test,Y_test


# In[23]:


def label2onehot(y,k=5):
    # y, 1-d array of long type
    N = len(y)
    return torch.zeros(N,k).scatter_(1,y.reshape(N,1),1)

def invsigmoid(x):
    return - torch.log(1/(x + 1e-8) - 1)

def recover_cp_pos(pred,L,n,v='v1'):
    '''
    pred : [n,], torch tnesor;
    when v=`v1`, L means L, n means `nc`;
    when v=`v2`, L means Ls, n means `ns`;
    '''
    pred = pred.unsqueeze(0) if pred.dim()==0 else pred
    n = len(pred)
    cp = []
    if v == 'v1':
        tmp = torch.round(pred*L).int()
        cp.append(tmp[0])
        for i in range(1,n):
            if tmp[i]<=tmp[i-1]:
                break
            cp.append(tmp[i])
    elif v == 'v2':
        cp.append(torch.round(pred[0]*L[0]).int())
        for i in range(1,n):
            cp.append(torch.round(pred[i]*L[i] + L[i-1]).int())
    else:
        raise ValueError('Unexpected v:',v)
    return cp


# In[18]:


def has_change_point(y):
    # y : [N,], 1-d array
    if len(y) == torch.sum(y==y[0]):
        return False
    else:
        return True
    
def find_change_points(y):
    # y : 1-d array
    #  change point is the point whose label is different from its previous one
    #  Point index starts from 1.
    label = y[0]
    pos = []
    if not has_change_point(y):
        return pos
    for i in range(1,len(y)):
        if label != y[i]:
            pos.append(i+1)
            label = y[i]
    return pos
    
def transform_pred(box_pred,L,num_cp,k):
    '''
    Decode box prediction into series prediction.
    Args:
        box_pred : [B,num_cp+(num_cp+1)*k];
        L : trip length;
        num_cp : number of change points in a trip;
        k : number of classes.
    Returns:
        y_pred : [B,L,k].
    '''
    B,_ = box_pred.shape
    pred = box_pred[:,num_cp:]
    pred = pred.reshape(B,num_cp+1,k)
    y_pred = torch.zeros(B,L,k)
    cp = torch.zeros(B,num_cp+1).int()
    cp[:,:num_cp] = torch.round(box_pred[:,:num_cp]*L).int()
    cp[:,num_cp] = L
    for i in range(B):
        if cp[i][0] == L:
            y_pred[i,:,:] = pred[i,0,:]
            continue
        inx_srt = 0
        for j in range(num_cp+1):
            if inx_srt >= cp[i][j]:
                y_pred[i,inx_srt:,:] = pred[i,j,:]
                break
            y_pred[i,inx_srt:cp[i][j],:] = pred[i,j,:]
            inx_srt = cp[i][j]
    return y_pred

def transform_pred_v2(box_pred,Ls,ns,k):
    '''
    Decode box prediction into series prediction.
    Args:
        box_pred : [B,(2k+1)*ns];
        Ls : length list, len(Ls) = `ns`,sum(Ls) = trip length;
        ns : number of segments in a trip;
        k : number of classes.
    Returns:
        y_pred : [B,L,k].
    '''
    B = box_pred.shape[0]
    pred = box_pred.reshape(B,ns,-1)
    L = np.sum(Ls)
    y_pred = torch.zeros(B,L,k)
    for i in range(B):
        srt = 0
        for j in range(ns):
            loc = torch.round(pred[i][j][-1]*Ls[j]).int()
#             loc = 4*torch.exp(invsigmoid(pred[i][j][-1]))
#             loc = min(loc.int(),Ls[j])
            y_pred[i,srt:(srt+loc),:] = pred[i,j,0:k]
            y_pred[i,(srt+loc):(srt+Ls[j]),:] = pred[i,j,k:(2*k)]
            srt += Ls[j]
    return y_pred


def get_cp_gt(y,num_cp,device="cpu"):
    # obtain ground truth of change points
    if len(y.shape) == 1:
        y = torch.unsqueeze(y,0)  # [L,] --> [1,L]
    B,L = y.shape
    cp_gt = torch.ones(B,num_cp).to(device)
    for i in range(B):
        y_tmp = y[i]
        y_tmp = y_tmp[y_tmp>=0]
        pos = find_change_points(y_tmp)
        for j in range(len(pos)):
            if j < num_cp:
                cp_gt[i][j] = pos[j]/L
    return cp_gt

def get_cp_cls_gt(y,num_cp,l):
    #  out cp_cls : [B,nc+1,k]
    B,L = y.size()
    cp_cls = torch.zeros(B,num_cp+1,k)
    for i in range(B):
        y_tmp = y[i][y[i]>=0]
        pos = find_change_points(y_tmp)
        cp_cls[i][0][y_tmp[0]] = 1
        for j in range(len(pos)):
            if j < self.num_cp:
                cp_cls[i][j][y_tmp[pos[j]-1]] = 1
    return cp_cls


# In[3]:


if __name__ == '__main__':
    cp = torch.Tensor([[0.9998, 0.9999]])
    k_pred = [[1.6480e-03, 9.9641e-01, 5.6183e-04, 3.4455e-04, 1.1986e-04],
            [1.7945e-02, 9.8975e-01, 1.9064e-03, 3.2295e-03, 2.0796e-03],
            [7.7133e-03, 9.8620e-01, 2.7227e-03, 2.3928e-03, 4.7887e-03]]
    k_pred = torch.Tensor(k_pred)
    k_pred = k_pred.reshape(1,15)
    pred = torch.cat((cp,k_pred),dim=1)
    y_pred = transform_pred(pred,400,2,5)
    print('y_pred:',y_pred.shape)


# In[16]:


if __name__ == '__main__':
    box_pred = torch.rand(2,3*(2*5+1))
    y_pred = transform_pred_v2(box_pred,[200,100,100],3,5)
    print('y_pred:',y_pred.shape)
    a = torch.argmax(y_pred[0],dim=1)


