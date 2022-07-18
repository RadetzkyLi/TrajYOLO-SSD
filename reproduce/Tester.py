#!/usr/bin/env python
# coding: utf-8

# In[20]:


import torch
from torch.autograd import Variable
import numpy as np
import time
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

import DeepCls


# In[ ]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python Tester.ipynb   ')
    except:
        pass


# In[2]:


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
        
def print_metrics(y_pred,y_true,**kwargs):
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


# In[39]:


def seg_to_point(y_pr,y_gt):
    '''
    Args:
        y_pr : [N,]
        y_gt : [N,num_modes]
    Returns:
        pr : [N',];
        gt : [N',].
    '''
    y_gt = np.array(y_gt,dtype=np.int)
    num = np.sum(y_gt)
    num_modes = y_gt.shape[1]
    pr,gt = np.zeros((num,),dtype=np.int),np.zeros((num,),dtype=np.int)
    inx = 0
    for i in range(len(y_pr)):
        n = np.sum(y_gt[i,:])
        pr[inx:(inx+n)] = y_pr[i]
        cnt = 0
        for j in range(num_modes):
            gt[(inx+cnt):(inx+cnt+y_gt[i,j])] = j
            cnt += y_gt[i,j]
        inx += n
    return pr,gt


# In[42]:


class DLTester:
    def __init__(self):
        pass
    
    def test_step(self,model,data):
        x,y = data
        pred = model(x)
        y_pred = pred.max(1)[1]
        return y_pred
    
    def test_model(self,X_test,Y_test,model,batch_size=16,num_classes=5,use_gpu = False,**kwargs):
        '''
        Evaluate performance of model on test set.
        :param X_test : torch tensor of size [B,L,C];
        :param Y_test : torch tensor of size [B,k];
        :param model : deep learning model that will be evaluated;
        :param batch_size : batch size for testing;
        :param num_classes : number of classes of transportation modes.
        '''
        def empty_cache():
            torch.cuda.empty_cache()

        n_samples = len(Y_test)
        num_batch = int(np.ceil(n_samples/batch_size))
        y_pred = None
        # get prediction of all test samples
        start_time = time.process_time()
        model.eval()
        X_test = Variable(torch.from_numpy(X_test).float())
        Y_test = Variable(torch.from_numpy(Y_test).float())
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
            pred = self.test_step(model,[x,y])
            if y_pred is None:
                y_pred = pred
            else:
                y_pred = torch.cat((y_pred,pred),dim=0)
        print('total',n_samples,'samples are tested,consuming time:',time.process_time()-start_time,'second.')
        # score by segment
        gt = torch.argmax(Y_test,dim=1).cpu()
        print_metrics(y_pred.cpu(),gt)
        # score by point
        print('\n score by point:')
        pr,gt = seg_to_point(y_pred.cpu(),Y_test.cpu().numpy())
        print_metrics(pr,gt)
        empty_cache()


# In[43]:


if __name__ == '__main__':
    B,L,C = 100,50,4
    X = torch.randn(B,L,C).float().numpy()
    Y = (torch.rand(B,5)*5).long().numpy()
    save_path = './test_trainer.pt'
    model = DeepCls.CNN_G(num_feats=C,L=L)
    model.load_state_dict(torch.load(save_path))
    tester = DLTester()
    tester.test_model(X,Y,model,batch_size=5)


# In[ ]:




