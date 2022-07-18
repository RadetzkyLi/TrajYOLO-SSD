import torch
import imp

import sys
sys.path.append('.')
from layers import models
from layers import Tester,MyLayers
from utils import util,extractor


if __name__ == "__main__":
    model_param_path = './weights/yolo_conv_1.pt' # your model param path
    num_classes = 5
    index_feats = [1,2,3]
    n_feats = len(index_feats)
    model = torch.load(model_param_path)
    # load test dataset
    data_path = './data/trips_fixed_len_400_8F.pickle'
    """ For deep learning based backbones under TrajYOLO or TrajSSD"""
    X_test,Y_test = util.load_seg_data(data_path,is_train=False)
    """ If backbone is hand-crafted method, write as follows:
    X_test,Y_test = extractor.load_linreg_data(data_path,is_train=False,norm_type='minmax')
    """
    print('Test set:',X_test.size(),Y_test.size())
    tester = Tester.DLTester(cp_v='v1')
    tester.test_model(X_test,Y_test,model,
                      batch_size=64,
                      num_classes=num_classes,
                      thd=150,
#                       cm_path = './data/cm_yolo_conv.csv'
                      use_gpu = True,
                      index_feats = index_feats
                     )