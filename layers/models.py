#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[4]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python models.ipynb   ')
    except:
        pass


# In[7]:


class PointNetFeat(nn.Module):
    def __init__(self,
                 n_feats = 4,
                 dims_up = [64,128],
                 only_global = True
                ):
        super(PointNetFeat,self).__init__()
        self.only_global = only_global
        # raise dimension
        dims_conv = [n_feats] + dims_up
        self.conv_embed = self.get_conv(dims_conv)
        self.bn_embed = self.get_bn(dims_up)

        self.last_dim = dims_up[-1]
        self.relu = nn.ReLU()
    
    def get_conv(self,dims,scales=None):
        conv = []
        if scales is None:
            for i in range(1,len(dims)):
                conv.append(nn.Conv1d(dims[i-1],dims[i],1))
        else:
            for i in range(1,len(dims)):
                conv.append(NewConv1d(dims[i-1],dims[i],scales[i-1],stride=1,padding=self.padding))
        return nn.ModuleList(conv)
            
    def get_bn(self,dims):
        bn = [nn.BatchNorm1d(dims[i]) for i in range(len(dims))]
        return nn.ModuleList(bn)
    
    def forward(self,x):
        #  x: [B,C,L]
        n_pts = x.size()[2]
        for i in range(len(self.conv_embed)):
            x = self.relu(self.bn_embed[i](self.conv_embed[i](x)))
#         for i in range(len(self.conv)):
#             x = self.relu(self.bn_conv[i](self.conv[i](x)))
        point_feat = x
        if self.only_global:
            dim = x.size()[1]
            x = torch.max(x,2,keepdim=False)[0]
            return x  #[B,Cout]
        # global features
        dim = x.size()[1]
        x = torch.max(x,2,keepdim=True)[0]
        x = x.reshape(-1,dim)
        x = x.view(-1,dim,1).repeat(1,1,n_pts)
        return torch.cat([point_feat,x],1)





class TrajYolo(nn.Module):
    def __init__(self,
                  n_feats = 3,
                  k = 5,
                  nc = 2, # number of change points
                  p_dropout = 0.5,
                  dims_down = [256,128],
                  feat = None
                 ):
        '''
        Args:
            feat : class for extract features,turn [B,C,L] into [B,C_out],which must has
                field `last_dim`, i.e., the dimension of output.
        '''
        super(TrajYolo,self).__init__()
        self.k = k
        self.nc = nc
        self.feat = PointNetFeat(n_feats) if feat is None else feat
        last_dim = nc + k*(nc+1)
        dims_fc = [self.feat.last_dim] + dims_down + [last_dim]
        self.fc = self.get_fc(dims_fc)
        self.dropout = self.get_dropout(dims_down,p_dropout)
        self.bn = self.get_bn(dims_down)
        self.relu = nn.ReLU()
        
    def get_fc(self,dims):
        fc = []
        for i in range(1,len(dims)):
            # fc.append(nn.Conv1d(dims[i-1],dims[i],1))
            fc.append(nn.Linear(dims[i-1],dims[i]))
        return nn.ModuleList(fc)
    
    def get_bn(self,dims):
        bn = [nn.BatchNorm1d(dims[i]) for i in range(len(dims))]
        return nn.ModuleList(bn)
    
    def get_dropout(self,dims,p):
        dropout = [nn.Dropout(p = p) for i in range(len(dims))]
        return nn.ModuleList(dropout)
    
    def forward(self,x):
        #  [B,L,C] --> [B,nc+(nc+1)*k]
        if len(x.shape) == 3:
            batch_size = x.shape[0]
            n_pts = x.shape[1]
            x = x.transpose(1,2)  # [B,n_feats,n_pts]
            x = self.feat(x)   # [B,C_out]
        else:
            x = self.feat(x)  #  the hand crafted features
        for i in range(len(self.bn)):
            x = self.relu(self.bn[i](self.fc[i](x)))
            x = self.dropout[i](x)
        x = self.fc[-1](x)
        x = torch.sigmoid(x)
       # x = x.reshape(batch_size,self.nc,self.k+2)
        return x


# In[27]:


if __name__ == '__main__':
    import MyLayers
    sim_data = torch.autograd.Variable(torch.rand(16,400,4))
    feat = MyLayers.ConvFeat(4,feat_type='yolo',dims_conv=[64,128,256],
                             scales_conv=[3,3,3],scales_pooling=[2,2,50],strides=[1,1,1])
    out = feat(sim_data.transpose(1,2))
    print(out.shape)
    net = TrajYolo(4,k=5,nc=2,feat = feat)
    out = net(sim_data)
    print(out.size())


class TrajSSD(nn.Module):
    def __init__(self,n_feats=3,k=5,L=400,feat=None):
        super(TrajSSD,self).__init__()
        self.L = L
        self.feat = feat
        self.conv = nn.Conv1d(feat.last_dim,k,kernel_size=1,stride=1,padding=0)
    
    def transform_pred(self,x,L):
        # [B,k,ns] --> [B,L,k]
        B,k,ns = x.shape
        x_new = torch.zeros(B,k,L).to(x.device)
        num = L // ns
        for i in range(ns):
            tmp = x[:,:,i].reshape(B,k,1)
            x_new[:,:,i*num:(i*num+num)] = tmp
        if num*ns < L:
            x_new[:,:,:ns*num] = x[:,:,-1].reshape(B,k,1)
        return x_new.transpose(1,2)
    
    def forward(self,x):
        # x : [B,L,C] --> [B,k,ns] --> [B,L,C]
        x = x.transpose(1,2)
        x = self.feat(x)  # [B,C',L']
        x = self.conv(x)
        x = x.sigmoid()
        x = self.transform_pred(x,self.L)
        return x


# In[6]:


if __name__ == '__main__':
    import MyLayers
    sim_data = torch.rand(20,400,3)
    feat = MyLayers.ResNetFeat(n_feats=3,feat_type='ssd')
    net = TrajSSD(n_feats=3,feat=feat)
    out = net(sim_data)
    print(out.shape)


# In[ ]:




