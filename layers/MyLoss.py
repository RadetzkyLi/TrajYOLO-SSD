#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn

import sys
sys.path.append('..')
from utils import util


# In[34]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python MyLoss.ipynb   ')
    except:
        pass


# In[12]:


class BaseLoss(nn.Module):
    def __init__(self,
                 k,
                 num_cp, # number of change points
                 weights = None,
                 reduction = None,
                 device = "cpu"
                ):
        super(BaseLoss,self).__init__()
        self.k = k
        self.num_cp = num_cp
        self.reduction = reduction
        self.device = device
    
    def get_cp_gt(self,y):
        B,L = y.size()
        cp_gt = torch.ones(B,self.num_cp)
        cp_gt = cp_gt.to(self.device)
        for i in range(B):
            y_tmp = y[i]
            y_tmp = y_tmp[y_tmp>=0]
            N = torch.sum(y_tmp>=0)
            pos = util.find_change_points(y_tmp)
            for j in range(len(pos)):
                if j < self.num_cp:
                    cp_gt[i][j] = pos[j]/L
#                     cp_gt[i][j] = pos[j]/N
        return cp_gt
    
    def get_cp_cls_gt(self,y):
        #  out cp_cls : [B,nc+1,k]
        B,L = y.size()
        cp_cls = torch.zeros(B,self.num_cp+1,self.k)
        cp_cls = cp_cls.to(self.device)
        for i in range(B):
            y_tmp = y[i][y[i]>=0]
            pos = util.find_change_points(y_tmp)
            cp_cls[i][0][y_tmp[0]] = 1
            for j in range(len(pos)):
                if j < self.num_cp:
                    cp_cls[i][j][y_tmp[pos[j]-1]] = 1
        return cp_cls
    
    def get_cp_masking(self,y,cp_gt=None):
        # y : [B,L]
        if cp_gt != None:
            masking = cp_gt < 1
            masking = masking.to(self.device)
            return masking
        B,L = y.size()
        masking = torch.ones(B,self.num_cp) < 0
        masking = masking.to(self.device)
        for i in range(B):
            y_tmp = y[i]
            y_tmp = y_tmp[y_tmp>=0]
            pos = util.find_change_points(y_tmp)
            j = min(len(pos),self.num_cp)
            masking[i][:j] = True
        return masking
    
    def get_nc_weight(self,cp_gt,y):
        '''use number of change points for weighting'''
        _,L = y.size()
        w = torch.sum(cp_gt<1,dim=1,keepdim=True) + 1
        w = w.reshape(-1,1,1)
        w = w.repeat(1,L,self.k)
        w[w>1] = 4.15
        w = w[y>=0]  # w : [B,L,k]-->[N,k]
        return w


# In[18]:


class RegLoss(BaseLoss):
    def __init__(self,
                 k = 5,
                 num_cp = 2,
                 device = "cpu",
                 reduction = None,
                 w_cp = 1,  #  weight of change point
                 w_cls = 1,  #  weight of classification
                 weight = None,
                 nc_w = False, # use number of change point for weighting
                 gamma = 2,
                 loss_type = 'sse',
                 transform = 'yolo_cp',
                 ns = 3, # number of segments in a trip
                 Ls = [200,100,100]
                ):
        super(RegLoss,self).__init__(k=k,num_cp=num_cp,device=device,reduction=reduction)
        self.w_cp = w_cp
        self.w_cls = w_cls
        self.loss_type = loss_type
        self.weight = weight
        self.nc_w = nc_w
        self.transform = transform
        self.ns = ns
        self.Ls = Ls
        self.BCELoss = torch.nn.BCELoss(weight=weight,reduction='sum')
        self.FocalLoss = FocalLoss(k=k,gamma=gamma,device=device)
        
    def transform_input(self,box_pred,y):
        '''
        Transform input so as to calculate loss.
        Args:
            box_pred : box-like prediction of shape (B,nc + (nc+1)*k) under framework TrajYOLO,
                else of shape (B,L,k) under framework TrajSSD;
            y : ground truth of labels, of shape (B,L).
        Returns:
            cp_gt : ground truth of change points, [B,nc];
            cp_pred : prediction of change points, [B,nc];
            y_gt : ground truth of each point, one-hot, [N,k];
            y_pred : prediction of each point, [N,k].
        '''
        cp_gt = self.get_cp_gt(y)
        B,L = y.size()
        if self.transform == 'yolo_cp':
            cp_pred = box_pred[:,:self.num_cp]
            y_pred = util.transform_pred(box_pred.cpu(),L,self.num_cp,self.k)
        elif self.transform == 'yolo_ns':
            cp_pred = box_pred[:,:self.num_cp]
            y_pred = util.transform_pred_v2(box_pred.cpu(),self.Ls,self.ns,self.k)
        elif self.transform == 'ssd':
            y_pred = box_pred
            cp_pred = self.get_cp_gt(torch.max(y_pred,dim=-1))
        masking = y >= 0
        y_pred = y_pred[masking]
        y_pred = y_pred.to(self.device)
        if self.nc_w:
            w = self.get_nc_weight(cp_gt,y)
            y_pred = torch.mul(w,y_pred)
        y = y[masking]
        y = self.label2onehot(y)
        y = y.to(self.device)
        return cp_gt,cp_pred,y,y_pred
    
    def label2onehot(self,y):
        N = len(y)
        return torch.zeros(N,self.k).to(self.device).scatter_(1,y.reshape(N,1),1)
    
    def get_cp_gt(self,y):
        if self.cp_gt == None:
            return super().get_cp_gt(y)
        return self.cp_gt
        
    def SSE(self,y_pred,y):
        return torch.sum(torch.pow(y-y_pred,2))
    
    def isigmoid(self,x):
        return - torch.log(1/(x + 1e-8) - 1)
    
    def compute_cp_loss(self,pr,gt,kind='ratio'):
        if kind == 'ratio':
            return torch.pow(pr-gt,2)
        elif kind == 'exp':
            pr_ori = self.isigmoid(pr)
            gt_ori = self.isigmoid(gt)
            a,L = 4,400
            err = torch.pow(torch.log(L/a*pr_ori) - torch.log(L/a*gt_ori),2)
            return err
        else:
            raise ValueError('Unexpected kind:',kind)
    
    def loss_bce(self,y_pred,y):
        _,_,y_gt,y_pred = self.transform_input(box_pred,y)
        return self.BCELoss(y_pred,y_gt)
    
    def loss_focal(self,box_pred,y):
        _,_,y_gt,y_pred = self.transform_input(box_pred,y)
        return self.FocalLoss(y_pred,y_gt)
    
    def loss_ori_yolo(self,box_pred,y):
        # cp_gt : [B,nc], cp_cls_gt : [B,nc+1,k]
        B = box_pred.size()[0]
        cp_gt = self.get_cp_gt(y)
        cp_pred = box_pred[:,:self.num_cp]
        cp_masking = self.get_cp_masking(y)
        loss_cp = torch.pow(cp_gt - cp_pred,2)
        loss_cp = torch.sum(loss_cp[cp_masking])
        cp_cls_gt = self.get_cp_cls_gt(y)
        cp_cls_pred = box_pred[:,self.num_cp:]
        cp_cls_pred = cp_cls_pred.reshape(B,self.num_cp+1,self.k)
        cls_masking = torch.ones(B,1) > 0
        cls_masking = cls_masking.to(self.device)
        cls_masking = torch.cat((cls_masking,cp_masking),1)
        loss_cls = torch.pow(cp_cls_gt - cp_cls_pred,2)
        loss_cls = torch.sum(loss_cls[cls_masking])
        return self.w_cp * loss_cp + self.w_cls * loss_cls
    
    def loss_sse(self,box_pred,y):
        # Sum Square Error Loss
        cp_gt,cp_pred,y_gt,y_pred = self.transform_input(box_pred,y)
        loss_cp = torch.sum(torch.pow(cp_gt - cp_pred,2))
        err = torch.pow(y_pred - y_gt,2)
        if self.nc_w:
            w = self.get_nc_weight(cp_gt,y)
            err = torch.mul(w,err)
        loss_cls = torch.sum(err)
        loss = self.w_cp * loss_cp + self.w_cls * loss_cls
        return loss
    
    def loss_sse_v1(self,box_pred,y):
        # Sum Square Error Loss with mask
        cp_gt,cp_pred,y_gt,y_pred = self.transform_input(box_pred,y)
        cp_masking = self.get_cp_masking(y,cp_gt)
        tmp = torch.pow(cp_gt - cp_pred,2)
        loss_cp = torch.sum(tmp[cp_masking])
        err = torch.pow(y_pred - y_gt,2)
        if self.nc_w:
            w = self.get_nc_weight(cp_gt,y)
            err = torch.mul(w,err)
        loss_cls = torch.sum(err)
        loss = self.w_cp * loss_cp + self.w_cls * loss_cls
        return loss
    
    def forward(self,box_pred,y,cp_gt=None):
        self.cp_gt = cp_gt
        if self.loss_type == 'sse':
            return self.loss_sse(box_pred,y)
        elif self.loss_type == 'sse_v1':
            return self.loss_sse_v1(box_pred,y)
        elif self.loss_type == 'bce':
            return self.loss_bce(box_pred,y)
        elif self.loss_type == 'focal':
            return self.loss_focal(box_pred,y)
        elif self.loss_type == 'ori_yolo':
            return self.loss_ori_yolo(box_pred,y)
        else:
            raise ValueError('Unexpected loss type:',self.loss_type)


# In[19]:


if __name__ == '__main__':
    B = 20
    nc = 2
    k = 5
    L = 10
    box_pred = torch.rand((B,nc + (nc+1)*k))
    y = torch.rand((B,L))*k
    y = y.long()
    y[0:5,:5] = 1
    y[0:5,5:] = -1
    loss_func = RegLoss(k,nc,loss_type='focal',nc_w=False)
    loss = loss_func(box_pred,y)
    print(loss)


class FocalLoss(nn.Module):
    def __init__(self,
                 k,
                 alpha = 0.5,
                 gamma = 2,
                 device = "cpu"
                ):
        super(FocalLoss,self).__init__()
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        
    def forward(self,pred,target):
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        pred = torch.cat((1-pred,pred),dim=1)
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).to(self.device)
        class_mask.scatter_(1,target.long(),1.)
        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs.clamp(min=0.0001,max=1.0)
        log_p = probs.log()
        #  remedy sample imbalance
        alpha = torch.ones(pred.shape[0],pred.shape[1]).to(self.device)
        alpha[:,0] = alpha[:,0] * (1 - self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)
        #  calculate loss
        batch_loss = -alpha*(torch.pow(1-probs,self.gamma))*log_p
        return batch_loss.sum()


# In[8]:


if __name__ == '__main__':
    pred = torch.rand(20,1)
    y = torch.round(torch.rand(20,1)).long()
    func = FocalLoss(k=5,gamma=2)
    loss = func(pred,y)
    print(loss)
