#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import DeepCls


# In[8]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python Trainer.ipynb   ')
    except:
        pass


# In[2]:


def my_metrics(y_pred,y):
    acc = accuracy_score(y,y_pred)
    return acc

def plot_loss_or_acc(arr_train,arr_val,item):
    if item == 'acc':
        y1 = arr_train 
        y2 = arr_val
    elif item == 'loss':
        y1 = arr_train
        y2 = arr_val
    else:
        raise ValueError('Unexpected item:',item)
    epochs = range(len(y1))
    plt.plot(epochs,y1,'r')
    plt.plot(epochs,y2,'g')
    plt.title("Training and Validation "+item)
    plt.xlabel('epochs')
    plt.ylabel(item)
    plt.legend(['train','validation'])
    #plt.show()
    
def plot_loss_and_acc(loss_train,loss_val,acc_train,acc_val):
    plt.figure(1,figsize=(10,5))
    plt.subplot(1,2,1)
    plot_loss_or_acc(loss_train,loss_val,'loss')
    
    plt.figure(1)
    plt.subplot(1,2,2)
    plot_loss_or_acc(acc_train,acc_val,'acc')


# In[7]:


class DLTrainer:
    '''
    A class for training deep learning networks.
    '''
    def __init__(self,model,num_classes,lr=0.001,save_path='model.pt',with_y=False):
        self.model = model
        self.lr = lr
        self.lr_list = []
        self.num_classes = num_classes
        self.save_path = save_path
        self.with_y = with_y # if True, y of [N,k], else y of [N,]
        self.optimizer = torch.optim.Adam(model.parameters(),lr = lr)
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        # early stopping
        self.patience = None
        self.best_score = None
        self.early_stop_cnt = 0
        self.val_loss_min = np.Inf
    
    def format_second(self,seconds):
        if seconds < 1:
            return str(seconds)+"s"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return '%02d:%02d:%02d'%(h,m,s)
    
    def validate_step(self,data):
        x,y = data
        if self.with_y: # y of [B,K]
            pred = self.model(x,y)
            y = y.max(1)[1]
            loss = self.loss_func(pred,y.max(1)[1])
        else:  # y of [B,]
            pred = self.model(x)
            loss = self.loss_func(pred,y)
        y_pred = pred.max(1)[1]
        return y_pred,loss.item()
        
    def validate(self,data,batch_size):
        '''
        Evaluate performance of model on validation set.
        
        '''
        x,y = data
        n_samples = len(y)
        num_batch = int(np.ceil(n_samples/batch_size))
        y_pred = None
        val_loss = 0
        for k in range(num_batch):
            if k == num_batch-1:
                inx_end = min((k+1) * batch_size,n_samples)
            else:
                inx_end = (k+1) * batch_size
            inx_srt = k * batch_size
            data = [x[inx_srt:inx_end],y[inx_srt:inx_end]]
            y_pred_batch,batch_loss = self.validate_step(data)
            val_loss += batch_loss
            if y_pred is None:
                y_pred = y_pred_batch 
            else:
                y_pred = torch.cat([y_pred,y_pred_batch],dim=0)
        self.val_loss.append(val_loss/len(y))
        self.val_acc.append(my_metrics(y_pred.cpu(),y.cpu()))
        
    def early_stop(self,val_loss,min_delta):
        score = - val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + min_delta:
            self.early_stop_cnt += 1
            if self.early_stop_cnt >= self.patience:
                print("Training early stopping because val_loss didn't decrease ",                      min_delta,'for ',self.patience,'epochs.')
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,save_weights_only=self.save_weights_only)
            self.early_stop_cnt = 0
        return False
    
    def save_checkpoint(self,val_loss=None,save_weights_only=True):
        '''
        Save model when validation loss decrease.
        '''
        if val_loss is None:
            print('save model to ',self.save_path)
        else:
            print(f'Validation loss decreased ({self.val_loss_min:.6f}-->{val_loss:.6f})            ,saving model to {self.save_path}')
            self.val_loss_min = val_loss
        if save_weights_only:
            torch.save(self.model.state_dict(),self.save_path)
        else:
            torch.save(self.model,self.save_path)
        
    def train_step(self,data):
        x,y = data
        self.optimizer.zero_grad()
        if self.with_y:
            pred = self.model(x,y)
            y = y.max(1)[1]
            loss = self.loss_func(pred,y)
        else:
            pred = self.model(x) # [B,k] for classification
            loss = self.loss_func(pred,y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(),y,pred.max(1)[1]
    
    def train(self,data,batch_size,verbose=0):
        X,Y = data
        n_samples = len(Y)
        num_batch = int(np.ceil(n_samples/batch_size))
        loss_epoch ,N_valid = 0,0
        Y_valid,Y_pred_valid = None,None
        for k in range(num_batch):
            if k == num_batch-1:
                inx_end = min((k+1)*batch_size,n_samples)
            else:
                inx_end = (k+1)*batch_size
            inx_srt = k*batch_size
            if inx_srt >= inx_end - 1:
                continue    # drop out batch containing only one sample.
            y = Y[inx_srt:inx_end]
            x = X[inx_srt:inx_end]  # [B,n_pts,n_features]

            loss_batch,y_valid,y_pred_valid = self.train_step([x,y])
            loss_epoch += loss_batch
            N_valid += len(y_valid)
            if Y_valid is None:
                Y_valid = y_valid
                Y_pred_valid = y_pred_valid
            else:
                Y_valid = torch.cat((Y_valid,y_valid),dim=0)
                Y_pred_valid = torch.cat((Y_pred_valid,y_pred_valid),dim=0)
                
            if verbose == 1:
                print('\r batches: {:d}/{:d}'.format(k+1,num_batch),end='',flush=True)
                
        # save train loss and acc
        self.loss.append(loss_epoch/N_valid)
        if self.use_gpu:
            Y_pred_valid = Y_pred_valid.cpu()
            Y_valid = Y_valid.cpu()
        self.acc.append(my_metrics(Y_pred_valid,Y_valid))
        
    def show_training_process(self,verbose=1,delta_time=None):
        if verbose != 1:
            return
        if self.patience is not None:
            print('\n[%d/%d] - loss:%f - acc:%f - val_loss:%f - val_acc:%f - %s'%                  (self.cur_epoch,self.epochs,self.loss[-1],self.acc[-1],
                   self.val_loss[-1],self.val_acc[-1],self.format_second(delta_time)))
        else:
             print('\n[%d/%d] loss:%f - acc:%f - %s'%                      (self.cur_epoch,self.epochs,self.loss[-1],self.acc[-1],self.format_second(delta_time)))
                
    def plot_loss_acc(self):
        plot_loss_and_acc(self.loss,self.val_loss,self.acc,self.val_acc)
    
    def scheduler_step(self):
        if self.scheduler is None:
            return
        self.scheduler.step()
        self.lr_list.append(self.scheduler.get_last_lr[0])
        
    def fit(self,X_train,Y_train,X_val=None,Y_val=None,
            batch_size = 32,
            n_epochs = 30,
            patience = None, 
            min_delta = 0.005 ,
            loss_func = torch.nn.NLLLoss(reduction='sum'),
            scheduler = None,
            verbose = 1,
            use_gpu = False,
            save_weights_only = True,
            device = 'cpu',
            **kwargs):
        '''
        Train network given parameters.
        :param X_train : tensor of size [samples,maxlen,n_features] or [samples,timesteps,features]
        :param Y_train : tensor of size [samples] or [samples,timesteps]
        :param patience :must be None or an positive interger,standing for 
            that if loss don't drop min_delta after early_stopping epochs ,then we
            stop training.If the param is not None,then X_val and Y_val must be not 
            None.
        
        '''
        if patience is not None and (X_val is None or Y_val is None):
            raise ValueError('When patience is not None,X_val and Y_val must be not None either.')
        self.batch_size = batch_size
        self.patience = patience
        self.loss_func = loss_func
        self.epochs = n_epochs
        self.scheduler = scheduler
        self.save_weights_only = save_weights_only
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device(device if self.use_gpu else "cpu")
        
        start_time1 = time.process_time()
        self.model.to(self.device)
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        for epoch in range(n_epochs):
            start_time2 = time.process_time()
            # set model to training mode
            self.cur_epoch = epoch + 1
            self.model.train()
            self.train([X_train,Y_train],self.batch_size,verbose)
            
            if patience is None:
                delta_time = time.process_time()-start_time2
                self.show_training_process(verbose=verbose,delta_time=delta_time)
                continue
            # set model to evaluation mode
            if self.use_gpu:
                X_val = X_val.to(self.device)
                Y_val = Y_val.to(self.device)
            self.model.eval()
            self.validate([X_val,Y_val],self.batch_size)
            
            # schedule learing rate
            self.scheduler_step()
            
            delta_time = time.process_time()-start_time2
            self.show_training_process(verbose=verbose,delta_time=delta_time)
            
            if self.early_stop(self.val_loss[-1],min_delta):
                break
        print('%d epochs for training finished - %s'%
              (n_epochs,self.format_second(time.process_time()-start_time1)))
        # save model after all epochs if didn't set early stopping.
        if patience is None:
            self.save_checkpoint(save_weights_only = self.save_weights_only)


# In[5]:


if __name__ == '__main__':
    B,L,C = 10,50,4
    n = 6
    X = torch.randn(B,L,C).float()
    Y = (torch.rand(B)*5).long()
    save_path = './test_trainer.pt'
    model = DeepCls.CNN_G(num_feats=C,L=L)
    #model = DeepCls.LSTM_P(num_feats=C)
    trainer = DLTrainer(model,5,save_path=save_path)
    trainer.fit(X[:n,:,:],Y[:n],X[n:,:,:],Y[n:],
                batch_size = 2,
                n_epochs = 5,
                patience = 1
               )


# In[ ]:




