#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# In[48]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python MyLayers.ipynb   ')
    except:
        pass


# # 0. Basic functions

# In[2]:


def get_conv(dims,scales=None,strides=None):
    # [B,Cin,L] --> [B,Cout,L]
    conv = []
    for i in range(1,len(dims)):
        scale = 1 if scales == None else scales[i-1]
        padding = 0 if scale == 1 else scale//2
        stride = 1 if strides == None else strides[i-1]
        conv.append(nn.Conv1d(dims[i-1],dims[i],kernel_size=scale,stride=stride,padding=padding))
    return nn.ModuleList(conv)

def get_bn(dims):
    bn = [nn.BatchNorm1d(dims[i]) for i in range(len(dims))]
    return nn.ModuleList(bn)

def get_ln(normalized_shapes):
    return [nn.LayerNorm(shape) for shape in normalized_shapes]

def get_maxpool(scales):
    maxpool = [nn.MaxPool1d(kernel_size=scales[i],padding=0) for i in range(len(scales))]
    return maxpool


# In[3]:


class MHSA(nn.Module):
    '''
    Implementation of Multi-head Self-Attention.
    Ref : https://blog.csdn.net/weixin_48167570/article/details/123832394
    '''
    def __init__(self,d_model,nhead):
        super().__init__()
        # Q, K, V transform matrices, we assume channels remain unchnaged.
        self.q = nn.Linear(d_model, d_model,bias=False)
        self.k = nn.Linear(d_model, d_model,bias=False)
        self.v = nn.Linear(d_model, d_model,bias=False)
        self.nhead = nhead
        
    def forward(self, x):
        B, N, C = x.shape
        # muti-head
        q = self.q(x).reshape(B, N, self.nhead, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.nhead, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.nhead, -1).permute(0, 2, 1, 3)
        
        # calculate attention score by dot product
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # multiply attention score and then output
        v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return v


# In[21]:


if __name__ == '__main__':
    net = MHSA(512,2)
    data = torch.rand(16,400,512)
    out = net(data)
    print(out.shape)


# # 1. Extract feats

# In[33]:


class MSPointNetFeat(nn.Module):
    def __init__(self,
                 n_feats = 3,
                 dims_up = [64,128],
                 dims_ms = [256,512],
                 scales_conv = [1,1],
                 pooling = False,
                 L = 400  # length of a trip
                ):
        """When `scales_conv` are [1,1,...], the model is equivalent to MLP"""
        super(MSPointNetFeat,self).__init__()
        self.pooling = pooling
        # raise dimension
        dims_conv = [n_feats] + dims_up
        self.conv_embed = self.get_conv(dims_conv)
        self.bn_embed = self.get_bn(dims_up)
        # conv
        dims_conv = [dims_up[-1]] + dims_ms
        self.conv = self.get_conv(dims_conv,scales_conv)
        self.bn_conv = self.get_bn(dims_ms)
        
        self.last_dim = dims_ms[-1] if pooling else dims_ms[-1]*L
        self.relu = nn.ReLU()
        
    def get_conv(self,dims,scales=None):
        return get_conv(dims,scales)
            
    def get_bn(self,dims):
        return get_bn(dims)
    
    def get_maxpool(self,scales):
        return get_maxpool(scale)
    
    def forward(self,x):
        # x : [B,C,L]
        n_pts = x.size()[2]
        for i in range(len(self.conv_embed)):
            x = self.relu(self.bn_embed[i](self.conv_embed[i](x)))
        for i in range(len(self.conv)):
            x = self.relu(self.bn_conv[i](self.conv[i](x)))
        if self.pooling:
            # global features
            dim = x.size()[1]
            x = torch.max(x,2,keepdim=True)[0]
            x = x.reshape(-1,dim)
        else:
            x = x.reshape(x.shape[0],-1)
        return x


# In[36]:


if __name__ == '__main__':
    sim_data = torch.autograd.Variable(torch.rand(16,4,50))
    point_feat = MSPointNetFeat(dims_up = [64,128],
                                dims_ms = [256],
                                scales_conv = [3],
                                pooling = False,
                                L = 50
                               )
    out = point_feat(sim_data)
    print('point feat:',out.shape)
    print(point_feat.last_dim)


# In[54]:


class ConvFeat(nn.Module):
    def __init__(self,
                 n_feats = 3,
                 dims_conv = [64,128,256,512,1024],
                 scales_conv = [3,3,3,3,3],
                 scales_pooling = [2,2,2,2,2],
                 strides = [1,1,1,1,1],
                 L = 400,
                 feat_type = 'yolo'  #  'yolo' or 'ssd'
                ):
        super(ConvFeat,self).__init__()
        self.feat_type = feat_type
        self.scales_pooling = scales_pooling
        self.bn = self.get_bn(dims_conv)
        dims_conv = [n_feats] + dims_conv
        self.conv = self.get_conv(dims_conv,scales_conv,strides)
        self.maxpool = self.get_maxpool(scales_pooling)
        
        self.last_dim = self.calc_last_dim(L,dims_conv,scales_pooling,strides)
        self.relu = nn.ReLU()
    
    def get_conv(self,dims,scales=None,strides=None):
        # [B,Cin,L] --> [B,Cout,L]
        return get_conv(dims,scales,strides)
        
    def get_bn(self,dims):
        return get_bn(dims)
    
    def get_maxpool(self,scales):
        return get_maxpool(scales)
    
    def calc_last_dim(self,L,dims_conv,scales_pooling,strides):
        if self.feat_type == 'ssd':
            return dims_conv[-1]
        L_out = L
        for i in range(len(strides)):
            L_out = L_out // strides[i]
            L_out = L_out // scales_pooling[i]
        return L_out*dims_conv[-1]
    
    def forward(self,x):
        #  x : [B,C,L]  --> [B,C_out]
        for i in range(len(self.conv)):
            x = self.relu(self.bn[i](self.conv[i](x)))
            if self.scales_pooling[i] > 1:
                x = self.maxpool[i](x)
        if self.feat_type == 'yolo':
            return torch.flatten(x,start_dim=1)
        elif self.feat_type == 'ssd':
            return x
        else:
            raise NotImplemented


# In[56]:


if __name__ == '__main__':
    sim_data = torch.autograd.Variable(torch.rand(16,4,400))
    feat = ConvFeat(n_feats = 4,
                    dims_conv = [64,128,256,512,1024],
                    scales_conv = [3,3,3,3,3],
                    scales_pooling = [2,2,2,2,2],
                    strides = [1,1,1,1,2],
                    L = 400,
                    feat_type = 'ssd'
                   )
    print(feat.last_dim)
    out = feat(sim_data)
    print('point feat:',out.shape,feat.last_dim)


# In[3]:


class ResNetBlock(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 kernel_size = 3
                ):
        super(ResNetBlock,self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        padding = kernel_size // 2
        if in_filters == out_filters:
            self.conv1 = nn.Conv1d(in_filters,out_filters,kernel_size,stride=1,padding=padding)
            self.conv2 = nn.Conv1d(out_filters,out_filters,kernel_size,stride=1,padding=padding)
        else:
            #  we assume that every time dimension doubles then size halves.
            self.conv1 = nn.Conv1d(in_filters,out_filters,kernel_size,stride=2,padding=padding)
            self.conv2 = nn.Conv1d(out_filters,out_filters,kernel_size,stride=1,padding=padding)
            self.conv3 = nn.Conv1d(in_filters,out_filters,kernel_size,stride=2,padding=padding)  # shortcut path
        self.bn1 = nn.BatchNorm1d(in_filters)
        self.bn2 = nn.BatchNorm1d(out_filters)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # x : [B,C,L] --> [B,C,L_out]
        if self.in_filters == self.out_filters:
            res_x = self.relu(self.bn2(self.conv1(x)))
            res_x = self.bn2(self.conv2(res_x))
            return self.relu(res_x+x)
        else:
            res_x = self.relu(self.bn2(self.conv1(x)))
            res_x = self.bn2(self.conv2(res_x))
            return self.relu(res_x + self.bn2(self.conv3(x)))


# In[6]:


if __name__ == '__main__':
    sim_data = torch.rand(20,4,10)
    res_block = ResNetBlock(4,6)
    print(sim_data.size())
    out = res_block(sim_data)
    print(out.size())


# In[50]:


class ResNetFeat(nn.Module):
    def __init__(self,
                 n_feats = 3,
                 dims_up = [64,128],
                 scales_up = [3,3],
                 scales_pooling = [2,2],
                 dims_resnet = [128,256,512],
                 L = 400,
                 pool = 'max',
                 feat_type = 'yolo'
                ):
        super(ResNetFeat,self).__init__()
        #  embedding
        self.bn_up = get_bn(dims_up)
        dims = [n_feats] + dims_up
        self.conv_up = get_conv(dims,scales_up)
        self.maxpool = get_maxpool(scales_pooling)
        #  resnet
        dims = [dims_up[-1]] + dims_resnet
        self.resnet = self.get_resnet(dims)
        #  params
        self.pool = pool
        self.feat_type = feat_type
        #  other params
        self.relu = nn.ReLU()
        self.last_dim = self.calc_last_dim(L,dims_resnet,scales_pooling)
    
    def calc_last_dim(self,L,dims,scales_pooling):
        if self.feat_type == 'ssd' or self.pool == 'avg' or self.pool == 'max':
            return dims[-1]
        L_out = L
        for i in range(len(scales_pooling)):
            L_out = L_out // scales_pooling[i]
        for i in range(1,len(dims)):
            if dims[i]>dims[i-1]:
                L_out = int(np.ceil(L_out/2))
        return L_out*dims[-1]
    
    def get_resnet(self,dims):
        res = []
        for i in range(1,len(dims)):
            res.append(ResNetBlock(dims[i-1],dims[i]))
        return nn.ModuleList(res)
    
    def forward(self,x):
        '''
        feat_type == 'yolo': x, [B,C,L] --> [B,C'];
        feat_type == 'ssd' : x, [B,C,L] --> [B,C',L'];
        '''
        for i in range(len(self.conv_up)):
            x = self.relu(self.bn_up[i](self.conv_up[i](x)))
            x = self.maxpool[i](x)
        for net in self.resnet:
            x = net(x)
        if self.feat_type == 'ssd':
            return x
        elif self.feat_type == 'yolo':
            if self.pool == 'max':
                x = torch.max(x,2,keepdim=True)[0]
            elif self.pool == 'avg':
                x = torch.mean(x,2)
            x = x.reshape(-1,self.last_dim)
            return x
        else:
            raise NotImplemented


# In[53]:


if __name__ == '__main__':
    sim_data = torch.rand(20,3,400)
    net = ResNetFeat(n_feats = 3,
                     dims_up = [32,64,128],
                     scales_up = [3,3,3],
                     scales_pooling = [2,2,2],
                     dims_resnet = [128,256],
                     pool = 'no',
                     feat_type = 'yolo',
                     L = 400
                    )
    out = net(sim_data)
    print(net.last_dim,out.size())


# In[3]:


class LSTMFeat(nn.Module):
    def __init__(self,
                 n_feats = 3,
                 dims = [64,128,256,512],
                 shortcut = False,
                 bidirectional = False
                ):
        super(LSTMFeat,self).__init__()
        dims_lstm = [n_feats] + dims
        self.lstm = self.get_lstm(dims_lstm,bidirectional)
        self.shortcut = shortcut
        self.bidirectional = bidirectional
        self.bn = get_bn(dims)
        self.relu = nn.ReLU()
        self.last_dim = dims[-1] if bidirectional else dims[-1]
        
    def get_lstm(self,dims,bidirectional):
        lstm = []
        for i in range(1,len(dims)):
            hidden_size = dims[i]//2 if bidirectional else dims[i]
            lstm.append(nn.LSTM(input_size=dims[i-1],hidden_size=hidden_size,num_layers=1,
                                batch_first=True,bidirectional=bidirectional))
        return nn.ModuleList(lstm)
    
    def forward(self,x):
        # [B,C,L] --> [B,C_out]
        x = x.transpose(1,2)  # [B,L,C]
        for i in range(len(self.lstm)-1):
            x_lstm,_ = self.lstm[i](x)
            if i >=2 and self.shortcut and not self.bidirectional:
                x_lstm = x_lstm + x
            x = x_lstm  # self.relu(x_lstm)
        _,(x,_) = self.lstm[-1](x)
        x = x.transpose(0,1)  # [D*num_layers,B,H_out] --> [B,D*layers,H_out]
        x = x.reshape(x.shape[0],-1)
        return x


# In[5]:


if __name__ == '__main__':
    sim_data = torch.rand(20,3,10)
    net = LSTMFeat(n_feats=3,dims=[128,512],bidirectional=True,shortcut=True)
    out = net(sim_data)
    print(out.size())


# # 2. Get prediction

# In[26]:
class FullConvFeat_v1(nn.Module):
    def __init__(self,
                 n_feats = 3,
                 k = 5,
                 L = 400,
                 dims = [64,128,256,512],
                 scales = [3,3,7,7],
                 strides = [1,1,1,1],
                 pools = [2,2,2,2],
                 feat_type = 'global',
                 extra = 'no',
                 dim_extra = 128,
                 norm = 'bn'
                ):
        super(FullConvFeat_v1,self).__init__()
        self.pools = pools
        self.feat_type = feat_type
        self.extra = extra
        self.norm = norm
        self.valid_extras = ['lstm','bi_lstm','transformer','mhsa']
        
        self.bn = get_bn(dims)
        self.ln = get_ln(dims)
        self.maxpools = get_maxpool(pools)
        self.conv = get_conv([n_feats]+dims,scales,strides)
        self.last_dim = self.calc_last_dim(feat_type,dims[-1],dim_extra)
        
        self.extra_nn = self.build_extra_nn(dim_extra,dims[-1])
        self.maxpool = nn.MaxPool1d(3,stride=1,padding=1)
        self.relu = nn.ReLU()
        
    def calc_last_dim(self,feat_type,dim_conv,dim_extra):
        if feat_type == 'local' or feat_type == 'global':
            dim = dim_conv*2
        elif feat_type == 'local_global':
            dim = dim_conv*3
        else:
            dim = dim_conv
        if self.extra == 'lstm':
            return dim+dim_extra
        elif self.extra == 'bi_lstm':
            return dim+2*dim_extra
        elif self.extra == 'transformer':
            return dim + dim_conv
        elif self.extra == 'mhsa':
            return dim + dim_conv
        return dim
        
    def build_extra_nn(self,dim_extra,dim_input):
        if self.extra == 'lstm':
            return nn.LSTM(input_size=dim_input,hidden_size=dim_extra,num_layers=1,batch_first=True)
        elif self.extra == 'bi_lstm':
            return nn.LSTM(input_size=dim_input,hidden_size=dim_extra,num_layers=1,batch_first=True,bidirectional=True)
        elif self.extra == 'transformer':
            return nn.TransformerEncoderLayer(d_model=dim_input, nhead=dim_extra,dim_feedforward=2*dim_input)
        elif self.extra == 'mhsa':
            return MHSA(dim_input,dim_extra)
        else:
            return None
            
    def forward_extra(self,x,ns):
        # x : [B,C,L] --> [B,C',L']
        if self.extra == 'lstm' or self.extra == 'bi_lstm':
            _,(x,_) = self.extra_nn(x.transpose(1,2))
            x = x.transpose(0,1)
            x = x.reshape(x.shape[0],-1).unsqueeze(2)
            x = x.repeat(1,1,ns)
        elif self.extra == 'transformer':
            x = x.transpose(1,2).transpose(0,1)
            x = self.extra_nn(x)  # [L,B,C]
            x = x.transpose(0,1).transpose(1,2)
        elif self.extra == 'mhsa':
            x = self.extra_nn(x.transpose(1,2))
            x = x.transpose(1,2)
        return x
        
    def forward(self,x):
        # [B,C,L] --> [B,C',L']
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            if self.norm == 'ln':
                x = self.ln[i](x.transpose(1,2))
                x = self.relu(x.transpose(1,2))
            else:
                x = self.relu(self.bn[i](x))
            if self.pools[i]>1:
                x = self.maxpools[i](x)
        # global and local features
        _,_,ns = x.shape
        x_extra = self.forward_extra(x,ns)
        if self.feat_type == 'local':
            lf = self.maxpool(x)
            x = torch.cat((x,lf),dim=1)
        elif self.feat_type == 'global':
            gf = torch.max(x,2,keepdim=True)[0] # [B,C,1]
            x = torch.cat((x,gf.repeat(1,1,ns)),dim=1)
        elif self.feat_type == 'local_global':  # 3P-CNN
            lf = self.maxpool(x)
            gf = torch.max(x,2,keepdim=True)[0] # [B,C,1]
            x = torch.cat((x,lf,gf.repeat(1,1,ns)),dim=1)
        # add extra data
        if self.extra in self.valid_extras:
            x = torch.cat((x,x_extra),dim=1)
        return x

# In[10]:


if __name__ == '__main__':
    sim_data = torch.rand(16,3,400)
    net = FullConvFeat_v1(dims=[16,32,64,128],scales=[3,3,3,3],strides=[2,2,2,2],pools=[1,1,1,1],
                          feat_type='no',extra='mhsa',dim_extra=2,norm='bn')
    out = net(sim_data)
    print(out.shape,net.last_dim)


# In[ ]:




