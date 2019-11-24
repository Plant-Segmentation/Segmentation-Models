from easydict import EasyDict as edict
from datasets import plant
import os
import numpy as np

config = edict()
#
config.exp_name = 'plant-pspNet-'+plant.experiment
if config.exp_name.split('category')[-1] == '11':
    config.test_snapshot   = ''
    config.test_hist_fname = 'Hist_category_11.csv'
    config.test_csv_fname  = 'Portion_category_11.csv'
    config.test_save_cls   = ["2-callus","3-callus red","4-shoot","5-shoot red","6-leaf","7-shoot brown","8-callus brown","9-leaf brown","10-root"]
    config.test_pickle_key = 'semSeg'

    config.train_snapshot = ''
    config.train_weights  = [1] + [10]*4 + [50]*6

elif config.exp_name.split('category')[-1] is '7':
    config.test_snapshot   = ''
    config.test_hist_fname = 'Hist_category_7.csv'
    config.test_csv_fname  = 'Portion_category_7.csv'
    config.test_save_cls   = ["2-callus", "3-callus red", "4-shoot","5-shoot red", "6-leaf"]
    config.test_pickle_key = 'semSeg'

    config.train_snapshot = ''
    config.train_weights  = [1, 10, 10, 100, 100, 100, 100]

elif config.exp_name.split('category')[-1] is '5':
    config.test_snapshot   = 'epoch_15_loss_0.48872_acc_0.96409_acc-cls_0.79643_mean-iu_0.69285_fwavacc_0.93817_lr_0.0049325993.pth'
    config.test_hist_fname = 'Hist_category_5.csv'
    config.test_csv_fname  = 'Portion_category_5.csv'
    config.test_save_cls   = ["2-callus", "3-shoot","4-leaf"]
    config.test_pickle_key = 'semSeg'

    config.train_snapshot = 'epoch_15_loss_0.48872_acc_0.96409_acc-cls_0.79643_mean-iu_0.69285_fwavacc_0.93817_lr_0.0049325993.pth'
    config.train_weights  = [1, 1, 1, 2, 2]

elif config.exp_name.split('category')[-1] is '4':
    config.test_snapshot   = 'epoch_297_loss_1.05541_acc_0.87718_acc-cls_0.77511_mean-iu_0.68723_fwavacc_0.78836_lr_0.0000072919.pth'
    #'epoch_149_loss_0.44672_acc_0.88368_acc-cls_0.85710_mean-iu_0.74456_fwavacc_0.80279_lr_0.0000412537.pth'
    config.test_hist_fname = 'Hist_category_4.csv'
    config.test_csv_fname  = 'Portion_category_4.csv'
    config.test_save_cls   = ["2-callus", "3-shoot"]
    config.test_pickle_key = 'semSeg'

    config.kmeans_rgb_cen  = {"2-callus":np.asarray([[10.0, 10],  [10, 120],  [120, 10], [200, 100]]),\
                              "3-shoot": np.asarray([[10, 10],  [10, 120],  [120, 10], [200, 100]])}
    config.kmeans_lab_cen  = {"2-callus":np.asarray([[132.1,143.1], [121.7,172.6], [130.2, 160.1], [147.3, 158.2]]),\
                              "3-shoot":np.asarray([[125.6,141.1], [109.3,176.6], [116.9, 161.9], [141.4, 160.3]])}

    config.train_snapshot = '' #'epoch_297_loss_1.05541_acc_0.87718_acc-cls_0.77511_mean-iu_0.68723_fwavacc_0.78836_lr_0.0000072919.pth'
    config.train_weights  = [1, 5, 5, 10]

#
config.save_dir = '../../result/'

if not os.path.exists(config.save_dir):
	os.makedirs(config.save_dir)

config.test_hist_fname = os.path.join(config.save_dir, config.test_hist_fname)
config.test_csv_fname  = os.path.join(config.save_dir, config.test_csv_fname)
config.test_analysis_mode = ['portion-color']
config.test_csv_title  = ",".join(['imgName', 'clsName', 'overall portion', 'green portion', 'red portion', 'brown portion'])
#
