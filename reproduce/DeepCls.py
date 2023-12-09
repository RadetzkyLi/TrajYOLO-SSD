#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import extractor



class CNN_G(nn.Module):
    '''
    This is an implementation of CNN (model G) proposed in :
    Dabiri, Sina, and Kevin Heaslip. "Inferring transportation modes from GPS trajectories 
    using a convolutional neural network." Transportation research part C: 
    emerging technologies 86 (2018): 360-371.
    '''
    def __init__(self,
                 num_feats,
                 k = 5,
                 p = 0.5,
                 L = 200
                ):
        '''
        Args:
            num_feats : number of point-wise features;
            k : number of categories;
            p : dropout date;
            L : length of a trip.
        '''
        super(CNN_G,self).__init__()
        self.relu = nn.ReLU()
        # conv layers
        padding = 1
        self.conv1 = nn.Conv1d(num_feats,32,3,padding=padding)
        self.conv2 = nn.Conv1d(32,32,3,padding=padding)
        self.conv3 = nn.Conv1d(32,64,3,padding=padding)
        self.conv4 = nn.Conv1d(64,64,3,padding=padding)
        self.conv5 = nn.Conv1d(64,128,3,padding=padding)
        self.conv6 = nn.Conv1d(128,128,3,padding=padding)
        
        # FC layers
        dim = int((L//8)*128)  #  maxpool with size 2 for 3 times, so disivor = 2^3
        self.fc1 = nn.Linear(dim,int(dim/4))
        self.fc2 = nn.Linear(int(dim/4),k)
        
        # dropout layers
        self.dp = nn.Dropout(p)
        
        # max pooling layer
        self.mp = nn.MaxPool1d(2,padding=0)
        
    def forward(self,x):
        # x : [B,L,C]
        # out : [B,k]
        x = x.transpose(1,2)  # [B,C,L]
        x = self.mp(self.relu(self.conv2(self.relu(self.conv1(x)))))
        x = self.mp(self.relu(self.conv4(self.relu(self.conv3(x)))))
        x = self.mp(self.relu(self.conv6(self.relu(self.conv5(x)))))
        x = self.dp(x)
        x = torch.flatten(x,start_dim=1) # [B,C']
        x = self.dp(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)


# In[51]:


if __name__ == '__main__':
    L = 20
    net = CNN_G(4,L=L)
    sim_data = torch.autograd.Variable(torch.rand(16,L,4))
    out = net(sim_data)
    print(out.shape)


# In[14]:


class LSTM_P(nn.Module):
    '''
    This is an implentation of the proposed LSTM model in:
    Yu, J. J. Q. (2019). Travel mode identification with gps trajectories using wavelet 
    transform and deep learning. IEEE Transactions on Intelligent Transportation Systems, 1-11.
    '''
    def __init__(self,
                 num_feats,
                 k = 5,
                 dwt = None,
                 mask_seq = False,
                 device = 'cpu'
                ):
        super(LSTM_P,self).__init__()
        self.relu = nn.ReLU()
        self.dwt = dwt
        self.mask_seq = mask_seq
        self.lstm1 = nn.LSTM(num_feats,128,batch_first=True)
        self.lstm2 = nn.LSTM(128,512,batch_first=True)
        self.lstm3 = nn.LSTM(512,512,batch_first=True)
        self.lstm4 = nn.LSTM(512,512,batch_first=True)
        self.lstm5 = nn.LSTM(512,512,batch_first=True)
        self.lstm6 = nn.LSTM(512,128,batch_first=True)
        
        dim_dwt = 112
        dim = 128 if dwt is None else 128+dim_dwt
        self.fc1 = nn.Linear(dim,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,k)
        
        self.dp = nn.Dropout(0.3)
    
    def pack(self,x,lens):
        x = torch.nn.utils.rnn.pack_padded_sequence(x,lens,batch_first=True,enforce_sorted=False)
        return x
    
    def unpack(self,x):
        x,lens = torch.nn.utils.rnn.pad_packed_sequence(x,batch_first=True)
        return x,lens
        
    def add(self,x1,x2):
        if self.mask_seq:
            x1,lens = self.unpack(x1)
            x2,_ = self.unpack(x2)
            x1 = x1 + x2
            x1 = self.pack(x1,lens)
        else:
            x1 = x1 + x2
        return x1
        
    def forward(self,x,y=None):
        # x : [B,L,C]
        # y : [B,k] , create masking for seq if `mask_seq` is True
        if self.dwt is not None:
            dwt_feats = self.dwt(x.cpu(),y.cpu())
        if self.mask_seq:
            lens = torch.sum(y,dim=1).long()
            x = self.pack(x,lens)
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x_lstm,_ = self.lstm3(x)
        x = self.add(x_lstm,x) #x_lstm + x
        x_lstm,_ = self.lstm4(x)
        x = self.add(x_lstm,x)
        x_lstm,_ = self.lstm5(x)
        x = self.add(x_lstm,x)
        _,(x,_) = self.lstm6(x) # [1,B,C']
        x = x.squeeze(0) # [B,C']
        if self.dwt is not None:
            x = torch.cat((x,dwt_feats),dim=1)
            pass
        x = self.dp(self.relu(self.fc1(x)))
        x = self.dp(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)


# In[16]:


if __name__ == '__main__':
    # unit test
    net = LSTM_P(4,mask_seq=True)
    sim_data = torch.randn(16,40,4)
    sim_y = torch.rand(16,5) * 3
    sim_y = sim_y.long()
    out = net(sim_data,sim_y)
    print(out.shape)
    # with dwt
    sim_y = (torch.zeros(16,5)).long()
    sim_y[:,0] = 20
    dwt = extractor.DWTFeat(4,y_fmt='v1')
    net = LSTM_P(4,mask_seq=True,dwt=dwt)
    out = net(sim_data,sim_y)
    print(out.shape)


# In[ ]:




