#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
import random
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

import data
from layers import Tester
from utils import util


# In[105]:


from imp import reload
reload(data)


# In[33]:


if __name__ == '__main__':
    try:
        # this is shell comman!
        get_ipython().system('jupyter nbconvert --to python ClassicCls.ipynb   ')
    except:
        pass


# # 1. Basic Functions

# In[2]:


class IdentityMapping():
    def __init__(self):
        pass
    def transform(self,x):
        return x


# In[111]:


def acc_segment(y_pr,y_gt):
    gt = np.argmax(y_gt,axis=1)
    acc = np.sum( y_pr == gt ) / len(y_pr)
    return acc

def acc_point(y_pr,y_gt):
    pr,gt = seg_to_point(y_pr,y_gt)
    acc = np.sum(pr == gt)/len(pr)
    return acc

def seg_to_point(y_pr,y_gt):
    '''
    Args:
        y_pr : [N,]
        y_gt : [N,num_modes]
    Returns:
        pr : [N',];
        gt : [N',].
    '''
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

def seg_to_point_v1(y_pr,y_gt):
    # y_pr : [N,]
    # y_gt : [N,2], first column is label, second column is count.
    num = np.sum(y_gt[:,1])
    pr,gt = np.zeros((num,),dtype=np.int),np.zeros((num,),dtype=np.int)
    inx = 0
    for i in range(len(y_pr)):
        pr[inx:(inx+y_gt[i,1])] = y_pr[i]
        gt[inx:(inx+y_gt[i,1])] = y_gt[i,0]
        inx += y_gt[i,1]
    return pr,gt

def cls_report_point(y_pr,y_gt):
    # y_pr : [N,]
    # y_gt : [N,2]
    pr,gt = seg_to_point(y_pr,y_gt)
    target_names = ['walk','bike','bus','car','train']
    print(classification_report(gt,pr,target_names=target_names,digits=3))
    #Tester.print_metrics(pr,gt)

def grid_search_clf(clf,X_train,Y_train,X_test_list=[],Y_test_list=[],seg_list=[],scale=None):
    # X_test_list and seg_list must have the same number of elements
    # Training and test trips can be divided by various segmentation method.
    if scale == 'norm':
        scaler = preprocessing.StandardScaler().fit(X_train)
    elif scale == 'minmax':
        scaler = preprocessing.MinMaxScaler().fit(X_train)
    else:
        scaler = IdentityMapping()
    Y = np.argmax(Y_train,axis=1)
    fit = clf.fit(scaler.transform(X_train),Y)
    pr = fit.best_estimator_.predict(scaler.transform(X_train))
    print('optimal params value:',fit.best_params_)
    print('training:       acc_s:%.3f, acc_p:%.3f'%(acc_segment(pr,Y_train),acc_point(pr,Y_train)))
    for i in range(len(X_test_list)):
        pr = fit.best_estimator_.predict(scaler.transform(X_test_list[i]))
        acc_s,acc_p = acc_segment(pr,Y_test_list[i]),acc_point(pr,Y_test_list[i])
        print('seg:%10s, acc_s:%.3f, acc_p:%.3f'%(seg_list[i],acc_s,acc_p))
        #cls_report_point(pr,Y_test_gt)
    return fit

def grid_search_clf_v1(clf,X_train,Y_train,X_test,Y_test,scale=None):
    '''Here, both X_train ans X_test come from same segmentation method.
    If a segment contains more than one travel mode, the most frequent travel mode will be 
    set as ground truth of the segment.
    '''
    if scale == 'norm':
        scaler = preprocessing.StandardScaler().fit(X_train)
    elif scale == 'minmax':
        scaler = preprocessing.MinMaxScaler().fit(X_train)
    else:
        scaler = IdentityMapping()
    Y = np.argmax(Y_train,axis=1)
    fit = clf.fit(scaler.transform(X_train),Y)
    pr = fit.best_estimator_.predict(scaler.transform(X_train))
    print('optimal params value:',fit.best_params_)
    print('training: acc_s:%.3f, acc_p:%.3f'%(acc_segment(pr,Y_train),acc_point(pr,Y_train)))
    pr = fit.best_estimator_.predict(scaler.transform(X_test))
    acc_s,acc_p = acc_segment(pr,Y_test),acc_point(pr,Y_test)
    print('          acc_s:%.3f, acc_p:%.3f'%(acc_s,acc_p))
    return fit


# # 2. Evaluation 
# RF,DT,KNN,SVM,MLP

# ## 2.1 use 11 features 

# In[65]:


if __name__ == '__main__':
    # the 11 features should not be smoothed due to their definition
    feats_kind = 'hand_v1'
    path = '../../TransModeDetection/data/feats_maxlen_2300_8F_wos.pickle'
    X_train,X_test_gt,Y_train,Y_test_gt = data.load_cls_data(path,None,10,'label',feats_kind,only_test=False)
    print('train:',X_train.shape,Y_train.shape,'test (gt):',X_test_gt.shape,Y_test_gt.shape)
    X_test_uni,Y_test_uni = data.load_cls_data(path,None,10,'uniform',feats_kind,only_test=True)
    print('train (uniform):',X_test_uni.shape)
    X_test_ws,Y_test_ws = data.load_cls_data(path,None,10,'ws',feats_kind,only_test=True)
    print('test (ws):',X_test_ws.shape)
    X_test_walk,Y_test_walk = data.load_cls_data(path,None,10,'walk',feats_kind,only_test=True)
    print('test (walk):',X_test_walk.shape)
    
    # conbine inputs
    X_test_list = [X_test_gt,X_test_uni,X_test_ws,X_test_walk]
    Y_test_list = [Y_test_gt,Y_test_uni,Y_test_ws,Y_test_walk]
    seg_list = ['gt','uniform','ws','walk']


# In[77]:


if __name__ == '__main__':
    # Random Forest
    RF = RandomForestClassifier(random_state=7)
    params = {'n_estimators':[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125]}
    clf_rf = GridSearchCV(estimator=RF, param_grid=params, cv=5)
    fit_rf = grid_search_clf(clf_rf,X_train,Y_train,X_test_list,Y_test_list,seg_list)


# In[78]:


if __name__ == '__main__':
    # SVM 
    SVM = SVC(gamma='scale')
    parameters = {'C': [0.5, 1, 4, 7, 10, 13, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]}
    clf_svm = GridSearchCV(estimator=SVM, param_grid=parameters, cv=5)
    fit_svm = grid_search_clf(clf_svm,X_train,Y_train,X_test_list,Y_test_list,seg_list,scale='norm')


# In[79]:


if __name__ == '__main__':
    # Multilayer perceptron
    C = X_train.shape[1]
    MLP = MLPClassifier(early_stopping=True, hidden_layer_sizes=(2*C,))
    parameters = {'hidden_layer_sizes': [(2*C,),(2*C,4*C),(2*C,4*C,8*C), (2*C,4*C,8*C,16*C), (2*C,4*C,8*C,16*C,32*C)]}
    clf_mlp = GridSearchCV(estimator=MLP, param_grid=parameters, cv=5)
    fit_mlp = grid_search_clf(clf_mlp,X_train,Y_train,X_test_list,Y_test_list,seg_list,scale='norm')


# In[80]:


if __name__ == '__main__':
    # Decision Tree
    DT = DecisionTreeClassifier()
    parameters = {'max_depth': [1, 5, 10, 15, 20, 25, 30, 35, 40]}
    clf_dt = GridSearchCV(estimator=DT, param_grid=parameters, cv=5)
    fit_dt = grid_search_clf(clf_dt,X_train,Y_train,X_test_list,Y_test_list,seg_list)


# In[76]:


if __name__ == '__main__':
    # KNN
    KNN = KNeighborsClassifier()
    parameters = {'n_neighbors': [3, 5, 10, 15, 20, 25, 30, 35, 40]}
    clf_knn = GridSearchCV(estimator=KNN, param_grid=parameters, cv=5)
    fit_knn = grid_search_clf(clf_knn,X_train,Y_train,X_test_list,Y_test_list,seg_list,scale='norm')





# ## 2.2 same segmentation to train and test 

# In[119]:


def get_clf(kind='RF',C=5):
    if kind == 'RF':
        RF = RandomForestClassifier()
        params = {'n_estimators':[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125]}
        clf = GridSearchCV(estimator=RF, param_grid=params, cv=5)
    elif kind == 'SVM':
        SVM = SVC(gamma='scale')
        parameters = {'C': [0.5, 1, 4, 7, 10, 13, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]}
        clf = GridSearchCV(estimator=SVM, param_grid=parameters, cv=5)
    elif kind == 'MLP':
        MLP = MLPClassifier(early_stopping=True, hidden_layer_sizes=(2*C,))
        parameters = {'hidden_layer_sizes': [(2*C,),(2*C,4*C),(2*C,4*C,8*C), (2*C,4*C,8*C,16*C), (2*C,4*C,8*C,16*C,32*C)]}
        clf = GridSearchCV(estimator=MLP, param_grid=parameters, cv=5)
    elif kind == 'DT':
        DT = DecisionTreeClassifier()
        parameters = {'max_depth': [1, 5, 10, 15, 20, 25, 30, 35, 40]}
        clf = GridSearchCV(estimator=DT, param_grid=parameters, cv=5)
    elif kind == 'KNN':
        KNN = KNeighborsClassifier()
        parameters = {'n_neighbors': [3, 5, 10, 15, 20, 25, 30, 35, 40]}
        clf = GridSearchCV(estimator=KNN, param_grid=parameters, cv=5)
    else:
        raise ValueError('Unexpected clf kind:',kind)
    return clf


# In[106]:


if __name__ == "__main__":
    path = '../../TransModeDetection/data/feats_maxlen_2300.pickle'
    feats_kind = 'hand'
    #path = '../../TransModeDetection/data/feats_maxlen_2300_8F_wos.pickle'
    seg_list = ['label','uniform','ws','walk']
    seg = seg_list[1]  #  select segmentation method to be used
    X_train,X_test,Y_train,Y_test = data.load_cls_data(path,None,10,seg,feats_kind,only_test=False,seg_train=seg)
    print('train:',X_train.shape,Y_train.shape,'test (gt):',X_test.shape,Y_test.shape)
    # grid search
    kinds = ['RF','SVM','MLP','DT','KNN']
    need_scale = ['SVM','MLP','KNN']
    for i in range(len(kinds)):
        print('\nclassifier: ',kinds[i])
        scale = kinds[i] if kinds[i] in need_scale else None
        clf = get_clf(kinds[i])
        fit = grid_search_clf_v1(clf,X_train,Y_train,X_test,Y_test,scale=scale)

