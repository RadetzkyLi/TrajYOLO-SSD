#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVC
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator,BayesianEstimator
import time
import pickle

import sys
sys.path.append('..')
import data


# In[23]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python ClassidSeg.ipynb   ')
    except:
        pass


# # 0. Basic Function 

# In[2]:


def format_second(seconds):
    if seconds < 1:
        return str(seconds)+"s"
    if seconds < 86400:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return '%02d:%02d:%02d'%(h,m,s)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return '%dday %02d:%02d:%02d'%(d,h,m,s)

def save_as_pickle(file_path,data):
    with open(file_path,'wb') as f:
        pickle.dump(data,f)

def load_pickle(file_path):
    with open(file_path,'rb') as f:
        return pickle.load(f)


# # 1. Classifiers
# for pointwise classification

# In[3]:


def bayesianBeliefNetwork(X_train,est='mle',prior_type='BDeu',equivalent_sample_size=5,pseudo_counts={}):
    '''
    Desc: the epoch level transportation mode identification using Bayesian Belief Network;
    Ref: Feng, T., & Timmermans, H. J. (2013). Transportation mode recognition using GPS and accelerometer data. 
        Transportation Research Part C: Emerging Technologies, 37, 118-130.
    Rerurns:
        model: trained BNN
    '''
    model = BayesianNetwork([('AVGSPEED','MODE'),('MAXSPEED','MODE')])
    if est == 'mle':
        estimator = MaximumLikelihoodEstimator
        model.fit(X_train,estimator=estimator)
    elif est == 'bpe':
        estimator = BayesianEstimator
        if prior_type == 'BDeu':
            model.fit(X_train,estimator=estimator,prior_type=prior_type,equivalent_sample_size=equivalent_sample_size)
        elif prior_type == 'dirichlet':
            model.fit(X_train,estimator=estimator,prior_type=prior_type,pseudo_counts=pseudo_counts)
        else:
            model.fit(X_train,estimator=estimator,prior_type=prior_type)
    else:
        raise NotImplemented('est:',est)
    return model


# In[4]:


def RF(X_train,X_test):
    '''
    Desc: point-level transortation modes identificaiton using Random Forest.
    Ref: Prelipcean, A. C., Gidófalvi, G., & Susilo, Y. O. (2014). Mobility collector. 
        Journal of Location Based Services, 8(4), 229-255.
    Args:
        X_train: (N,6),the last is MODE;
        X_test: (N',6).
    '''
    RF = RandomForestClassifier()
    params = {'n_estimators':[35, 45, 55, 65, 75, 85, 95, 105, 115, 125]}
    clf = GridSearchCV(estimator=RF, param_grid=params, cv=10)
    fit = clf.fit(X_train[:,:5],X_train[:,5])
    pr = fit.best_estimator_.predict(X_train[:,:5])
    # result of training set
    res = classification_report(X_train[:,5],pr,digits=3,output_dict=True)
    print('optimal params value:',fit.best_params_)
    print('training:  acc:%.3f, f1:%.3f'%(res['accuracy'],res['weighted avg']['f1-score']))
    # result of test set
    pr = fit.best_estimator_.predict(X_test[:,:5])
    print(classification_report(X_test[:,5],pr,digits=3))
    return fit

def DT(X_train,X_test):
    '''
    Desc: point-level transportation modes identification using Decision Trss.
    Ref: Prelipcean, A. C., Gidófalvi, G., & Susilo, Y. O. (2014). Mobility collector. 
        Journal of Location Based Services, 8(4), 229-255.
    Args:
        X_train: (N,6), tha last column is MODE;
        X_test: (N',6);
    Returns:
        fit: trained classifier.
    '''
    DT = DecisionTreeClassifier()
    parameters = {'max_depth': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}
    clf = GridSearchCV(estimator=DT, param_grid=parameters, cv=10)
    fit = clf.fit(X_train[:,:5],X_train[:,5])
    pr = fit.best_estimator_.predict(X_train[:,:5])
    # result of training set
    res = classification_report(X_train[:,5],pr,digits=3,output_dict=True)
    print('optimal params value:',fit.best_params_)
    print('training:  acc:%.3f, f1:%.3f'%(res['accuracy'],res['weighted avg']['f1-score']))
    # result of test set
    pr = fit.best_estimator_.predict(X_test[:,:5])
    print(classification_report(X_test[:,5],pr,digits=3))
    return fit

def SVM(X_train,X_test):
    '''
    Ref: Prelipcean, A. C., Gidófalvi, G., & Susilo, Y. O. (2014). Mobility collector. 
        Journal of Location Based Services, 8(4), 229-255.
    Args:
        X_train: (N,6), tha last column is MODE;
        X_test: (N',6);
    Returns:
        fit: trained classifier.
    '''
    # normalize data
    scaler = preprocessing.StandardScaler().fit(X_train[:,:5])
    # SVM 
    SVM = SVC(gamma='scale')
    parameters = {'C': [0.5, 1, 4, 7, 10, 13, 16, 20, 25, 30, 35]}
    clf = GridSearchCV(estimator=SVM, param_grid=parameters, cv=10)
    fit = clf.fit(scaler.transform(X_train[:,:5]),X_train[:,5])
    pr = fit.best_estimator_.predict(scaler.transform(X_train[:,:5]))
    # result of training set
    res = classification_report(X_train[:,5],pr,digits=3,output_dict=True)
    print('optimal params value:',fit.best_params_)
    print('training:  acc:%.3f, f1:%.3f'%(res['accuracy'],res['weighted avg']['f1-score']))
    # result of test set
    pr = fit.best_estimator_.predict(scaler.transform(X_test[:,:5]))
    print(classification_report(X_test[:,5],pr,digits=3))
    return fit,scaler


# # 2. Post-process

# In[5]:


def post_process(y_pred,lens,method='method3',full=True,num_modes=5):
    '''
    Desc: post-processing method to impose consistency in a trip.
    Ref: Feng, T., & Timmermans, H. J. (2015). Enhanced Imputation of GPS Traces Forcing Full or
        Partial Consistency in Activity Travel Sequences: Comparison of Algorithms. Transportation
        Research Record, 2430(1), 20-27.
    Args:
        y_pred: (N,),prediction of test set;
        lens: length list of test dataset;
        method: `method1`, `method2` or `method2`;
        full: impose full consistency or partial consistency;
        num_modes: number of transportation modes.
    Returns:
        y_post: (N,), N denotes the total GPS points.
    '''
    def _calc_mode_freq(y):
        freqs,modes = [1],[y[0]]
        for i in range(1,len(y)):
            if y[i] == modes[-1]:
                freqs[-1] += 1
            else:
                freqs.append(1)
                modes.append(y[i])
        return freqs,modes
    y_post = np.zeros((len(y_pred),),dtype=np.int32)
    if full:
        # the whole trip belongs to one mode 
        if method == 'method1':
            srt = 0
            for length in lens:
                end = srt + length
                freqs,modes = _calc_mode_freq(y_pred[srt:end])
                max_inx = np.argmax(np.array(freqs))
                y_post[srt:end] = modes[max_inx]
                srt = end
        elif method == 'method2':
            srt = 0
            for length in lens:
                end = srt +length
                freqs = [np.sum(y_pred[srt:end]==i) for i in range(num_modes)]
                y_post[srt:end] = np.argmax(freqs)
                srt = end
        elif method == 'method3':
            srt = 0 
            cnt = 0
            cnt_if = [0,0,0]
            for length in lens:
                end = srt + length
                freqs,modes = _calc_mode_freq(y_pred[srt:end])
                freqs_new = np.array([np.sum(y_pred[srt:end]==i) for i in range(num_modes)])
                if len(modes) == 1:
                    y_post[srt:end] = modes[0]
                    cnt_if[0] += 1
                elif len(modes) == 2:
                    y_post[srt:end] = modes[0] if freqs[0]>freqs[1] else modes[1]
                    cnt_if[1] += 1
                else:
                    freqs.pop()
                    freqs.pop(0)
                    modes.pop()
                    modes.pop(0)
                    mode = modes[np.argmax(np.array(freqs))]
                    y_post[srt:end] = mode
                    cnt_if[2] += 1
                srt = end
                cnt += 1
        else:
            raise ValueError('Unexpected method:',method)
    else:
        # except walk,all other mode can be changed
        walk = 0
        srt = 0
        for length in lens:
            end = srt + length
            freqs,modes = _calc_mode_freq(y_pred[srt:end])
            trip_mode = walk # trip mode other than walk segment
            if method == 'method1':
                order = np.argsort(-np.array(freqs))
                trip_mode = walk
                for i in order:
                    if not (modes[i] == walk):
                        trip_mode = modes[i] 
                        break
            elif method == 'method2':
                freqs_new = np.array([np.sum(y_pred[srt:end]==i) for i in range(num_modes)])
                trip_mode = np.argmax(freqs_new)
            elif method == 'method3':
                if len(modes) == 1:
                    trip_mode = modes[0]
                elif len(modes) == 2:
                    if modes[0] != walk and modes[1] != walk:
                        trip_mode = modes[0] if freqs[0]>freqs[1] else modes[1]
                else:
                    order = np.argsort(-np.array(freqs))
                    for i in range(1,len(order)-1):
                        if modes[order[i]] != walk:
                            trip_mode = modes[order[i]]
                            break
            else:
                raise ValueError('Unexpected method:',method)
            # modify predictions
            mid1,mid2 = srt,srt
            for i in range(len(modes)):
                mid2 += freqs[i]
                y_post[mid1:mid2] = walk if modes[i] == walk else trip_mode
                mid1 = mid2
            srt = end
    return y_post

def test_postprocess(y_gt,y_pr,lens):
    fulls = [True,False]
    methods = ['method1','method2','method3']
    res = classification_report(y_gt,y_pr,digits=3,output_dict=True)
    print('original predicton: acc:',res['accuracy'],'F1:',res['weighted avg']['f1-score'])
    for full in fulls:
        for method in methods:
            y_post = post_process(y_pr,lens,method,full)
            res = classification_report(y_gt,y_post,digits=3,output_dict=True)
            tmp = 'full' if full else 'partial'
            print(tmp,'-',method,': acc:',res['accuracy'],'F1:',res['weighted avg']['f1-score'])


# # 3. Experiments

# In[6]:


if __name__ == '__main__':
    ''' Bayesian Belief Network '''
    BASIC_DIR = 'your_data_dir/'
    path = BASIC_DIR + 'feats_maxlen_2300_8F.pickle'
    X_train,X_test,lens = data.load_seg_data(path,T=60,slide='center')
    X_train = pd.DataFrame(X_train,columns=['AVGSPEED','MAXSPEED','MODE'])
    X_test = pd.DataFrame(X_test,columns=['AVGSPEED','MAXSPEED','MODE'])
    pred_data = X_test.drop(columns=['MODE'],axis=1)
    names = ['walk','bike','bus','car','train']
    print(X_train.shape,X_test.shape)
    # fit
    prior_types = ['BDeu','dirichlet','K2']
    est = ['mle','bpe']
    model = bayesianBeliefNetwork(X_train,est=est[0])
    # predict
    y_pred = model.predict(pred_data)
    # print
    res = classification_report(X_test.loc[:,'MODE'].values,y_pred.loc[:,'MODE'].values,target_names=names,digits=3)
    print(res)


# In[20]:


if __name__ == '__main__':
    save_as_pickle('../data/segmodel_BBN.pickle',model)
    save_as_pickle('../data/seg_Y_BBN.pickle',y_pred.values)


# In[137]:


if __name__ == '__main__':
    test_postprocess(X_test.loc[:,'MODE'].values,y_pred.loc[:,'MODE'].values,lens)


# In[ ]:





# In[60]:


if __name__ == '__main__':     
    '''Random Forest'''
    X_train1,X_test1,lens = data.load_seg_data(path,ftype='RF')
    print(X_train1.shape,X_test1.shape)
    start_time = time.process_time()
    rf = RF(X_train1,X_test1)
    end_time = time.process_time()
    print('Training using time:',format_second(end_time-start_time))


# In[138]:


if __name__ == '__main__':
    pr = rf.best_estimator_.predict(X_test1[:,:5])
    test_postprocess(X_test1[:,5],pr,lens)


# In[ ]:


if __name__ == '__main__':
    save_as_pickle('./data/segmodel_RF.pickle',rf)
    save_as_pickle('./data/seg_Y_RF.pickle',pr)


# In[143]:






