#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import time
import csv

import sys
sys.path.append('..')
from utils import util


# In[3]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python Tester.ipynb   ')
    except:
        pass


# # 1 Basic functions

# ## 1.1 save and print metrics 

# In[2]:
    
def cp_metrics1(cnt):
    # cnt :[TP,FP,FN]
    print('change point TP,FP,FN:',cnt)
    print('change point precision,recall:',cnt[0]/cnt[3],cnt[0]/cnt[2])


# In[3]:


def metrics(y_pred,y_true,method = 'accuracy',average=None):
    '''
    Caculate some criterion according to method, including 
    accuracy,precision,recall,f1_score.
    :param method : criterion method;
    :param average : required for multi-class targets,default is None,
        if None,scores of each class are returned;
        if micro,caculate metrics globally;
        if macro,caculate unweighted mean score of each label; 
    '''
    if method == 'accuracy':
        return accuracy_score(y_true,y_pred)
    elif method == 'precision':
        return precision_score(y_true,y_pred,average=average,labels=[0,1,2,3,4])
    elif method == 'recall':
        return recall_score(y_true,y_pred,average=average,labels=[0,1,2,3,4])
    elif method == 'f1_score':
        return f1_score(y_true,y_pred,average=average,labels=[0,1,2,3,4])
    elif method == 'confusion_matrix':
        return confusion_matrix(y_true,y_pred)
    else:
        raise ValueError('Unsupported evaluate method:',method)


# In[4]:


def save_metrics(acc,f1_avg,precision,recall,f1_score,cm,k,output_path):
    '''
    Save them as a csv for making confusion matrix in Word conviniently.
    '''
    columns = ['walk','bike','bus','car','train','recall','samples']
    table = np.zeros((k+2,k+2))
    table[0:k,0:k] = cm
    table[0:k,k] = recall
    table[0:k,k+1] = np.sum(cm,axis=1)
    table[k,0:k] = precision
    table[k+1,0:k] = f1_score
    table[k,k+1] = acc
    table[k+1,k+1] = f1_avg
    with open(output_path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(table)

def print_metrics(y_pred,y_true,cm_path=None,**kwargs):
    '''
    Print all kinds of cretirion of classification.
    :cm_path : path for saving comfusion matrix;
    '''
    acc = metrics(y_pred,y_true,'accuracy')
    precision = metrics(y_pred,y_true,'precision')
    recall = metrics(y_pred,y_true,'recall')
    f1_score = metrics(y_pred,y_true,'f1_score')
    f1_score_avg = metrics(y_pred,y_true,'f1_score','weighted')
    cm = metrics(y_pred,y_true,'confusion_matrix')
    print('   accuracy:',acc)
    print('weighted-f1:',f1_score_avg)
    print('  precision:',precision)
    print('     recall:',recall)
    print('   f1-score:',f1_score)
    print('confusion matrix:\n',cm)
    if cm_path is not None:
        save_metrics(acc,f1_score_avg,precision,recall,f1_score,cm,len(recall),cm_path)


# In[ ]:


def postprocess_save(pred,masking):
    path = './data/res_ymasking.pickle'
    util.save_as_pickle(path,[pred,masking])
    return pred


# ## 1.2 functions for test model

# In[3]:


def get_ypred(box_pred,y,nc,k):
    _,L = y.size()
    y_pred = util.transform_pred(box_pred,L,nc,k)  #[B,L,k]
    y_pred = y_pred[y>=0]  #[N,k]
    y_pred = y_pred.max(1)[1]
    return y_pred

def eval_cp_step(box_pred,y,nc,thd=10):
    if len(y.size()) == 1:
        y = torch.unsqueeze(y,0)  #  [L,] --> [1,L]
        box_pred = torch.unsqueeze(box_pred,0)
    B,L = y.size()
#     print(box_pred.size())
    sae = 0  # sum absolute error
    y_cp, y_cp_pred = [],[]
    for i in range(B):
        y_tmp = y[i][y[i]>=0]
        L_valid = len(y_tmp)
        cp_pos = util.find_change_points(y_tmp)
        cur_pos = -3
        for j in range(nc):
            pos = int(box_pred[i][j]*L)
            if (pos <= cur_pos+2) or (pos >= L_valid):
                break
            if j < len(cp_pos):
                y_cp.append(1)
                tmp = abs(pos - cp_pos[j])
                sae = sae + tmp
                y_cp_pred.append(1 if tmp<=thd else 0)
            else:
                y_cp.append(0)
                y_cp_pred.append(1)
    y_cp = torch.Tensor(y_cp).long()
    y_cp_pred = torch.Tensor(y_cp_pred).long()
    return sae,y_cp_pred,y_cp

def get_cp_cnt(y_pred,y,thd=10):
    '''
    Change points from pred.If a predicted change point (CP) and a true CP are
    within `thd` GPS points, it will be seemed as accurate.
    Args:
        y_pred : [N,];
        y : [B,L];
        thd : positive interger.
    '''
    if len(y.size()) == 1:
        y = torch.unsqueeze(y,0)  #  [L,] --> [1,L]
    B,L = y.size()
    cnt = [0,0,0,0] # [TP,FP,FN]
    srt = 0
    SAE = 0  #  sum absolute error
    masking = y>=0
    for k in range(B):
        cp_gt = util.find_change_points(y[k][masking[k]])
        cp_pred = util.find_change_points(y_pred[srt:(srt+torch.sum(masking[k]))])
        srt = srt + torch.sum(masking[k])
        num_gt,num_pred = len(cp_gt),len(cp_pred)
        for i in range(min(num_gt,num_pred)):
            if abs(cp_gt[i]-cp_pred[i])<= thd:
                cnt[0] = cnt[0] + 1
            else:
                cnt[1] = cnt[1] + 1
            SAE = SAE + abs(cp_gt[i]-cp_pred[i])
        if num_pred>=num_gt:
            cnt[3] = cnt[3] + num_pred - num_gt
        else:
            cnt[2] = cnt[2] + num_gt - num_pred
    return SAE,torch.Tensor(cnt)

def get_cp_cnt_v2(box_pred,y,n=2,Ls=None,thd=10):
    '''
    Change points from box_pred. If predicted and true CP are within `thd` GPS points,
    then the predicted CP is regarded as accurate.
    Args:
        box_pred : [B,m]
    when v=`v1`, L means L, n means `nc`;
    when v=`v2`, n means `ns`;
    '''
    if len(y.size()) == 1:
        y = torch.unsqueeze(y,0)  #  [L,] --> [1,L]
        box_pred = torch.unsqueeze(box_pred,0)
    B,L = y.size()
    cnt = [0,0,0,0] # [TP,FP,FN]
    SAE = 0  #  sum absolute error
    masking = y>=0
    for k in range(B):
        cp_gt = util.find_change_points(y[k][masking[k]])
        if Ls is None:
            cp_pred = util.recover_cp_pos(box_pred[k],L,n,'v1')
        else:
            tmp = box_pred[k].reshape(n,-1)
            cp_pred = util.recover_cp_pos(tmp[:,-1],Ls,n,'v2')
        num_gt,num_pred = len(cp_gt),len(cp_pred)
        for i in range(min(num_gt,num_pred)):
            if abs(cp_gt[i]-cp_pred[i])<= thd:
                cnt[0] = cnt[0] + 1
            else:
                cnt[1] = cnt[1] + 1
            SAE = SAE + abs(cp_gt[i]-cp_pred[i])
        if num_pred>=num_gt:
            cnt[3] = cnt[3] + num_pred - num_gt
        else:
            cnt[2] = cnt[2] + num_gt - num_pred
    return SAE,torch.Tensor(cnt)

def get_cp_cnt_v3(y_pred,y,thd_dist,x):
    '''
    Change points (CPs) from y_pred and y.
    Args:
        y_pred: [N,];
        y : [B,L];
        thd_dist : distance threshold for determining true predicted CP;
        x : [B,L,C], features of the trip, first channel is distance;
    Returns:
        SAE : sum absolute error between true CPs and predicted CPs;
        cnt : [4,].
    '''
    if len(y.size()) == 1:
        y = torch.unsqueeze(y,0)  #  [L,] --> [1,L]
    B,L = y.size()
    cnt = [0,0,0,0] # [TP,,No. of true CP, No. of predicted CP]
    srt = 0
    SAE = 0  #  sum absolute error
    masking = y>=0
    for k in range(B):
        cp_gt = util.find_change_points(y[k][masking[k]])
        cp_pr = util.find_change_points(y_pred[srt:(srt+torch.sum(masking[k]))])
        srt = srt + torch.sum(masking[k])
        num_gt,num_pr = len(cp_gt),len(cp_pr)
        flag = [0 for _ in range(num_pr)]
        for i in range(num_gt):
            if num_pr == 0:
                break
            dist_list = []
            for j in range(num_pr):
                if flag[j] == 1:
                    dist_list.append(thd_dist+100)
                    continue
                inx_srt = min(cp_gt[i],cp_pr[j])
                inx_end = max(cp_gt[i],cp_pr[j])
                dist = 0 if inx_srt==inx_end else np.sum(x[k,inx_srt:inx_end,0].cpu().numpy())
                dist_list.append(dist)
            inx = np.argmin(dist_list)
            if dist_list[inx] < thd_dist:
                cnt[0] += 1
                flag[inx] = 1
                SAE += dist_list[inx]
        cnt[2] = cnt[2] + num_gt
        cnt[3] = cnt[3] + num_pr
    return SAE,torch.Tensor(cnt)


class DLTester:
    def __init__(self,dwt_test=None,cp_v='v1'):
        '''
        dwt_test : Discrete Wavelet Transform features of test set;
        cp_v : format of outputs to find CP , `v1` if outputs are predictions, `v2` 
            if outputs are box pred.
        '''
        self.dwt_test = dwt_test
        self.cp_v = cp_v 
        
    def get_ypred(self,box_pred,y,model):
        _,L = y.size()
        if hasattr(model,'ns'):
            y_pred = util.transform_pred_v2(box_pred.cpu(),model.Ls,model.ns,model.k)
        else:
            y_pred = util.transform_pred(box_pred.cpu(),L,model.nc,model.k)  #[B,L,k]
        y_pred = y_pred[y>=0]  #[N,k]
        y_pred = y_pred.max(1)[1]
        return y_pred
    
    def get_cp_cnt(self,pred,box_pred,y,model,thd,x=None):
        '''when cp_v == `v2`, pred is in fact box_pred'''
        if self.cp_v == 'v1':
            if x is None:
                return get_cp_cnt(pred,y,thd)
            else:
                return get_cp_cnt_v3(pred,y,thd,x)
        else:
            if hasattr(model,'ns'):
                return get_cp_cnt_v2(box_pred,y,model.ns,model.Ls,thd)
            else:
                return get_cp_cnt_v2(box_pred,y,model.nc,None,thd)
    
    def test_step(self,model,data,dwt=None):
        x,y = data
        if len(y.size())==1:
            y = torch.unsqueeze(y,0)
            x = torch.unsqueeze(x,0)
        if dwt == None:
            box_pred = model(x)
        else:
            box_pred = model(x,dwt)
        y_pred = self.get_ypred(box_pred,y,model)
        return y_pred,box_pred

    def test_model(self,
                    X_test,
                    Y_test,
                    model,
                    batch_size=16,
                    num_classes=5,
                    thd = 10,
                    cm_path = None,
                    use_gpu = False,
                    index_feats = None,
                    **kwargs):
        '''
        Evaluate performance of model on test set. 
        Args:
            X_test : torch tensor of size [n_smaples,timesteps,n_features];
            Y_test : torch tensor of size [n_smaples,timesteps];
            model : deep learning model that will be evaluated;
            batch_size : batch size for testing;
            num_classes : number of classes of transportation modes;
            thd : if distance between true and predicted change point is within `thd`,
                then the prediction is regarded as true.
            post_process : You can pass a `post_process` function to execute additional 
                operations on predictions such as saving prediction results. This fucntion 
                receive two params, `Y_pred` of shape (n_pts,k) and `masking` of shape (n_pts,).
            index_feats : if not `None`, we judge predicted CP by distance, else by GPS points interval. 
        '''
        def empty_cache():
            torch.cuda.empty_cache()

        if 'post_process' in kwargs.keys():
            post_process = kwargs['post_process']
        else:
            post_process = None

        n_samples = len(Y_test)
        num_batch = int(np.ceil(n_samples/batch_size))
        y_pred = None
        y_cp,y_cp_pred = None,None
        loss_mae = 0
        cp_cnt = torch.zeros(4)
        # get prediction of all test samples
        start_time = time.process_time()
        model.eval()
        if use_gpu and torch.cuda.is_available():
            model = model.cuda()
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()
        for k in range(num_batch):
            if k == num_batch-1:
                inx_end = min((k+1) * batch_size,n_samples)
            else:
                inx_end = (k+1) * batch_size
            inx_srt = k * batch_size
            x = X_test[inx_srt:inx_end]
            y = Y_test[inx_srt:inx_end]
            dwt = None if self.dwt_test == None else self.dwt_test[inx_srt:inx_end]
            if index_feats is None:
                pred,box_pred = self.test_step(model,[x,y],dwt)
                sae,cnt = self.get_cp_cnt(pred,box_pred,y,model,thd)
            else:
                pred,box_pred = self.test_step(model,[x[:,:,index_feats],y],dwt)
                sae,cnt = self.get_cp_cnt(pred,box_pred,y,model,thd,x)
            loss_mae = loss_mae + sae
            cp_cnt = cp_cnt + cnt
            if y_pred is None:
                y_pred = pred
            else:
                y_pred = torch.cat((y_pred,pred),dim=0)   
        loss_mae = loss_mae / (cp_cnt[0]+cp_cnt[1])
        masking = Y_test>=0
        y_true = Y_test[masking]
        print('total',n_samples,'samples are tested,consuming time:',time.process_time()-start_time,'second.')
        if post_process is not None:
            y_pred = post_process(y_pred.cpu(),masking.cpu())
        else:
            y_pred = y_pred.cpu()
        if cm_path is None:
            print('change point MAE:',loss_mae)
            cp_metrics1(cp_cnt)
            print_metrics(y_pred,y_true.cpu())
        else:
            print_metrics(y_pred,y_true.cpu(),save_cm=True,cm_path=cm_path)
        empty_cache()


# In[ ]:




