import torch
import imp

import layers
from layers import models
from layers.models import PointNetYolo,TrajYolo,TrajYolo_v2,TrajYolo_v3
from layers.models import TrajSSD
from layers import Trainer
from layers import MyLoss,MyLayers
from utils import util,extractor


# # 1. Train models
# ## 1.1 Train TrajYOLO with various backbones
if __name__ == '__main__':
    num_classes = 5
    lr = 0.001
    index_feats = [1,2,3]  # speed,acceleration and jerk
    n_feats = len(index_feats)
    num_cp = 2
    data_path = './data/trips_fixed_len_400_8F.pickle'
    save_path = './weights/yolo_conv_1.pt'
    """ The `ConvFeat` can be replaced by other deep learning backbones, e.g., 
    ResNet,LSTM,PointNet from `MyLayers`. """
    feat = MyLayers.ConvFeat(n_feats = n_feats,
                             dims_conv = [64,128,256,512,1024],
                             scales_conv = [3,3,3,3,3],
                             scales_pooling = [2,2,2,2,2],
                             strides = [1,1,1,1,1],
                             feat_type = 'yolo',
                             L = 400
                            )
    model = models.TrajYolo(n_feats = n_feats,
                     k = num_classes,
                     nc = num_cp,
                     feat = feat,
                     p_dropout = 0.5,
                     dims_down = [256,128]
                    )
    
    # load training data set
    X_train,X_val,Y_train,Y_val = util.load_seg_data(data_path,is_train=True)
    X_train,X_val = X_train[:,:,index_feats],X_val[:,:,index_feats]
    cp_gt_train,cp_gt_val = util.get_cp_gt(Y_train,num_cp),util.get_cp_gt(Y_val,num_cp)
    ''' If backbone is hand-crafted method, then we should write as follows:
    feat = extractor.IdentityFeat(11)
    model = models.TrajYolo(n_feats = 5,
                            nc = nc,
                            k = num_classes,
                            feat = feat
                            )
    X_train,X_val,Y_train,Y_val = extractor.load_linreg_data(data_path,is_train=True,norm_type='minmax')
    cp_gt_train,cp_gt_val = util.get_cp_gt(Y_train,num_cp),util.get_cp_gt(Y_val,num_cp)
    '''
    print(X_train.size(),X_val.size(),Y_train.size(),Y_val.size(),cp_gt_train.size())
    device = "cuda:1"
    trainer = Trainer.DLTrainer(model,num_classes,lr = lr,save_path=save_path)
    loss_func = MyLoss.RegLoss(num_classes,
                               num_cp = num_cp,
                               device = device,
                               loss_type = 'sse',
                               w_cp = 300,
                               w_cls = 1
                              )
    scheduler = Trainer.get_scheduler(trainer.optimizer,
                                      policy = 'multi_step',
                                      milestones = [10,20,30,40],
                                      gamma = 0.1
                                     )
    
    # set params for training
    n_epochs = 100
    patience = 15
    # start train
    trainer.fit(X_train,Y_train,X_val,Y_val,
                batch_size = 128,
                n_epochs = n_epochs,
                loss_func = loss_func,
                loss_divider = 'trip',
                scheduler = scheduler,
                patience = patience,
                min_delta = 0.1,
                use_gpu = True,
                device = device,
                cp_gt_train = cp_gt_train,
                cp_gt_val = cp_gt_val,
                save_weights_only = False
               )
               


# ## 1.2 Train TrajSSD with CNN-based backbones
if __name__ == '__main__':
    k = 5
    lr = 0.001
    index_feats = [1,2,3]
    n_feats = len(index_feats)
    save_path = './weights/ssd_conv.pt'
    feat = MyLayers.FullConvFeat_v1(n_feats = n_feats,
                                     dims = [64,128,256,512],
                                     scales = [3,3,7,7],
                                     strides = [1,1,1,1],
                                     pools = [2,2,2,2],
                                     extra = 'no',
                                     dim_extra = 512,
                                     feat_type = 'local_global',
                                     norm = 'bn'
                                    )
    model = models.TrajSSD(n_feats=n_feats,feat=feat)
    trainer = Trainer.DLTrainer(model,k,lr = lr,save_path=save_path)
    # load training data set
    data_path = './data/trips_fixed_len_400_8F.pickle'
    X_train,X_val,Y_train,Y_val = util.load_seg_data(data_path,is_train=True)
    X_train,X_val = X_train[:,:,index_feats],X_val[:,:,index_feats]
    cp_gt_train,cp_gt_val = util.get_cp_gt(Y_train,2),util.get_cp_gt(Y_val,2)
    print(X_train.size(),X_val.size(),Y_train.size(),Y_val.size())
    device = "cuda:1"
    # in fact, w_cp = 0 also works well under TrajSSD
    loss_func = MyLoss.RegLoss(k=k,w_cp=300,transform='ssd',device=device)
    scheduler = Trainer.get_scheduler(trainer.optimizer,
                                      policy = 'multi_step',
                                      milestones = [10,20,30,40],
                                      gamma = 0.1
                                      )
    # start train
    trainer.fit(X_train,Y_train,X_val,Y_val,
                batch_size = 128,
                n_epochs = 100,
                loss_func = loss_func,
                loss_divider = 'point',
                scheduler = scheduler,
                patience = 15,
                min_delta = 0.001,
                use_gpu = True,
                device = device,
                cp_gt_train = cp_gt_train,
                cp_gt_val = cp_gt_val,
                save_weights_only = False
               )
               
    trainer.plot_loss_acc()