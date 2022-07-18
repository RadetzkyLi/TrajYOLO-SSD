#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.autograd import Variable
import numpy as np

import DeepCls
import data


# In[ ]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python train_test.ipynb   ')
    except:
        pass


# # 0. Basic functions

# In[ ]:


def load_data(path,maxlen=200,minlen=20,seg='label',only_test=False,val_ratio=None):
    if only_test:
        X_test,Y_test = data.load_cls_data(path,maxlen,minlen,seg,'deep',only_test=only_test,seg_train=seg)
        X_test,Y_test = Variable(torch.from_numpy(X_test).float()),Variable(torch.from_numpy(Y_test).long())
        return X_test,Y_test
    X_train,X_test,Y_train,Y_test = data.load_cls_data(path,maxlen,minlen,seg,'deep',only_test=only_test,seg_train=seg)
    X_train,X_test = Variable(torch.from_numpy(X_train).float()),Variable(torch.from_numpy(X_test).float())
    Y_train,Y_test = Variable(torch.from_numpy(Y_train).long()),Variable(torch.from_numpy(Y_test).long())
    if val_ratio is None:
        return X_train,X_test,Y_train,Y_test
    N = X_train.shape[0]
    N = int(N*(1-val_ratio))
    
    return X_train[:N,:,:],X_train[N:,:,:],X_test,Y_train[:N,:],Y_train[N:,:],Y_test


# In[ ]:


def ensemble_CNN_G(X_train,Y_train,X_test,Y_test):
    num_models = 7
    preds = None
    N,L,C = X_train.shape
    _,num_cls = Y_train.shape
    feats_index = [1,2,3,4]
    batch_size = 64
    device = 'cuda:0'
    print('train,test:',X_train.shape,X_test.shape)
    for i in range(0,num_models):
        np.random.seed((i + 1) * 2)
        Number = np.random.choice(N, size=N, replace=True, p=None)
        Ens_X_train = torch.zeros(N,L,C).float()
        Ens_Y_train = torch.zeros(N,num_cls).long()
        counter = 0
        for j in Number:
            Ens_X_train[counter, :, :] = X_train[j,:,:]
            Ens_Y_train[counter, :] = Y_train[j, :]
            counter += 1
        # train model
        model = DeepCls.CNN_G(num_feats=4,L=L)
        save_path = './cls_cnn_ens_' + str(i) + '.pt'
        trainer = Trainer.DLTrainer(model,5,save_path=save_path)
        trainer.fit(X_train[:,:,feats_index],Y_train.max(1)[1],
                    batch_size = batch_size,
                    n_epochs = 100,
                    patience = None,
                    shuffle = True,
                    verbose = 0,
                    use_gpu = True,
                    device = device)
        # test
        model.eval()
        num_batch = int(np.ceil(N/batch_size))
        pred = None
        for k in range(num_batch):
            if k == num_batch-1:
                inx_end = min((k+1) * batch_size,N)
            else:
                inx_end = (k+1) * batch_size
            inx_srt = k * batch_size
            with torch.no_grad():
                pred_batch = model(X_test[inx_srt:inx_end,:,feats_index].to(device))
            if pred is None:
                pred = pred_batch
            else:
                pred = torch.cat((pred,pred_batch),dim=0)
        if preds is None:
            preds = pred
        else:
            preds += pred
        # empty cache
        del model
        torch.cuda.empty_cache()
        time.sleep(5)
    # calc final result
    preds = preds / num_models
    preds = preds.max(1)[1]
    Tester.print_metrics(preds.cpu(),Y_test.max(1)[1].cpu())
     # score by point
    print('\n score by point:')
    pr,gt = Tester.seg_to_point(preds.cpu(),Y_test.cpu().numpy())
    Tester.print_metrics(pr,gt)


# In[ ]:


def eval_model(model,tester,path,maxlen=200,minlen=20,segs=['label'],use_gpu=True,feats_index=[1,2,3,4]):
    for seg in segs:
        X_test,Y_test = load_data(path,maxlen,minlen,seg,only_test=True)
        print('\nseg:',seg,X_test.shape)
        tester.test_model(X_test[:,:,feats_index],Y_test,model,batch_size=128,use_gpu=use_gpu)

def eval_ens_cnn(model,base_params_path,X_test,Y_test,batch_size=128,feats_index=[1,2,3,4],num_models=7):
    N,L,C = X_test.shape
    preds = None
    num_batch = int(np.ceil(N/batch_size))
    device = next(model.parameters()).device
    for i in range(num_models):
        path = base_params_path + '_' + str(i) + '.pt'
        model.load_state_dict(torch.load(path))
        model.eval()
        pred = None
        for k in range(num_batch):
            if k == num_batch-1:
                inx_end = min((k+1) * batch_size,N)
            else:
                inx_end = (k+1) * batch_size
            inx_srt = k * batch_size
            with torch.no_grad():
                pred_batch = model(X_test[inx_srt:inx_end,:,feats_index].to(device))
            if pred is None:
                pred = pred_batch
            else:
                pred = torch.cat((pred,pred_batch),dim=0)
        if preds is None:
            preds = pred
        else:
            preds += pred
    # clac average 
    preds = preds / num_models
    preds = preds.max(1)[1]
    Tester.print_metrics(preds.cpu(),Y_test.max(1)[1].cpu())
     # score by point
    print('\n score by point:')
    pr,gt = Tester.seg_to_point(preds.cpu(),Y_test.cpu().numpy())
    Tester.print_metrics(pr,gt)


# # 1. Training

# ## 1.1 train CNN_G 

# In[ ]:


if __name__ == '__main__':
    """ Train CNN_G. """
    B,L,C = 64,200,4
    feats_index = [1,2,3,4]  # S,A,J,BR
    path = './data/feats_maxlen_2300_8F.pickle'
    X_train,_,Y_train,_ = load_data(path,maxlen=L,minlen=20,seg='label')
    print('train:',X_train.shape,Y_train.shape)
    save_path = './cls_cnn.pt'
    model = DeepCls.CNN_G(num_feats=C,L=L)
    trainer = DLTrainer(model,5,save_path=save_path)
    trainer.fit(X[:,:,feats_index],Y,
                batch_size = 64,
                n_epochs = 62,
                patience = 10,
                min_delta = 0.001,
                use_gpu = True,
                device = 'cuda:0',
                save_weights_only = False
               )


# ## 1.2 train ensemble CNN_G 

# In[ ]:


if __name__ == '__main__':
     B,L,C = 64,200,4
    X_train,X_test,Y_train,Y_test = load_data(path,maxlen=L,minlen=20,seg='label')
    print('train:',X_train.shape,Y_train.shape)
    ensemble_CNN_G(X_train,Y_train,X_test,Y_test)


# ## 1.3 train LSTM_P 

# In[ ]:


if __name__ == '__main__':
    B,L,C = 64,200,4
    feats_index = [1,2,3,4]  # S,A,J,BR
    path = '.data/feats_maxlen_2300_8F.pickle'
    X_train,X_test,Y_train,Y_test = load_data(path,maxlen=L,minlen=16,seg='walk')
    print('train:',X_train.shape,Y_train.shape)
    print('test:',X_test.shape,Y_test.shape)
    
    save_path = './cls_lstm.pt'
    model = DeepCls.LSTM_P(num_feats=C,mask_seq=True)
    trainer = Trainer.DLTrainer(model,5,save_path=save_path,with_y=True)
    trainer.fit(X_train[:,:,feats_index],Y_train,
                batch_size = 128,
                n_epochs = 100,
                patience = None,
                min_delta = 0.001,
                use_gpu = True,
                device = 'cuda:0',
                save_weights_only = False
               )


# # 2. Test 

# In[ ]:


if __name__ == '__main__':
    #  test one model with different X_test obtained by various segmentation methods
    segs = ['label','uniform','ws','walk']
    save_path = './cls_lstm.pt'
    model = torch.load(save_path)
    eval_model(model,tester,path,L,16,segs=segs)


# In[ ]:


if __name__ == '__main__':
    # test ensemble CNN_G
    L = 200
    base_params_path = './cls_cnn_ens'
    segs = ['label','uniform','ws','walk']
    model = DeepCls.CNN_G(num_feats=C,L=L)
    for seg in segs:
        X_test_t,Y_test_t = load_data(path,L,10,seg,only_test=True)
        print('\nseg:',seg,X_test_t.shape)
        eval_ens_cnn(model,base_params_path,X_test_t,Y_test_t)

