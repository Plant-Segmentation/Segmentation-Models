from easydict import EasyDict as edict
from datasets import plant
import os

config = edict()
#
config.exp_name = 'plant-fcn8x-'+plant.experiment
if config.exp_name.split('category')[-1] == '11':
    config.test_snapshot   = 'epoch_137_loss_569394.81250_acc_0.96934_acc-cls_0.55636_mean-iu_0.49163_fwavacc_0.95009_lr_0.0001000000.pth'
    config.test_hist_fname = 'Hist_category_11.csv'
    config.test_csv_fname  = 'Portion_category_11.csv'
    config.test_save_cls   = ["2=callus","3-callus red","4-shoot","5-shoot red","6-leaf","7-shoot brown","8-callus brown","9-leaf brown","10-root"]
    config.test_pickle_key = 'semSeg'

    config.train_snapshot = 'epoch_137_loss_569394.81250_acc_0.96934_acc-cls_0.55636_mean-iu_0.49163_fwavacc_0.95009_lr_0.0001000000.pth'
    config.train_weights  = [1] + [10]*4 + [50]*6

elif config.exp_name.split('category')[-1] is '7':
    config.test_snapshot   = 'epoch_152_loss_1480101.12500_acc_0.96510_acc-cls_0.56052_mean-iu_0.45515_fwavacc_0.94340_lr_0.0001000000.pth'
    config.test_hist_fname = 'Hist_category_7.csv'
    config.test_csv_fname  = 'Portion_category_7.csv'
    config.test_save_cls   = ["2=callus", "3-callus red", "4-shoot","5-shoot red", "6-leaf"]
    config.test_pickle_key = 'semSeg'

    config.train_snapshot = ''
    config.train_weights  = [1, 10, 10, 100, 100, 100, 100]

elif config.exp_name.split('category')[-1] is '5':
    # 1 branch result.
    #config.test_snapshot   = 'epoch_62_loss_521967.71875_acc_0.96761_acc-cls_0.87932_mean-iu_0.73417_fwavacc_0.94578_lr_0.0001000000.pth'
    config.test_snapshot   = 'epoch_152_loss_367870.12500_acc_0.96899_acc-cls_0.62999_mean-iu_0.54418_fwavacc_0.94752_lr_0.0001000000.pth'
    config.test_hist_fname = 'Hist_category_5.csv'
    config.test_csv_fname  = 'Portion_category_5.csv'
    config.test_save_cls   = ["2=callus", "3-shoot","4-leaf"]
    config.test_pickle_key = 'semSeg'

    config.train_snapshot = ''
    config.train_weights  = [1, 10, 10, 50, 50]

elif config.exp_name.split('category')[-1] is '4':
    config.test_snapshot   = '276_loss_121088.13619_acc_0.89500_acc-cls_0.88493_mean-iu_0.73734_fwavacc_0.82102_lr_0.0001000000.pth'
    config.test_hist_fname = 'Hist_category_4.csv'
    config.test_csv_fname  = 'Portion_category_4.csv'
    config.test_save_cls   = ["2=callus", "3-shoot"]

    config.test_pickle_key = 'semSeg'

    config.train_snapshot = '276_loss_121088.13619_acc_0.89500_acc-cls_0.88493_mean-iu_0.73734_fwavacc_0.82102_lr_0.0001000000.pth'
    config.train_weights  = [1, 1, 1, 2]

elif config.exp_name.split('category')[-1] == '_color':
    config.test_snapshot   = 'epoch_324_loss_122699.88652_acc_0.91185_acc-cls_0.72810_mean-iu_0.60430_fwavacc_0.85669_lr_0.0000100000.pth'
    config.test_hist_fname = 'Hist_category_color.csv'
    config.test_csv_fname  = 'Portion_category_color.csv'
    config.test_save_cls   = ["1-green", "2=red", "3-brown"]

    config.test_pickle_key = 'colorSeg'

    config.train_snapshot = 'epoch_322_loss_75300.37986_acc_0.91152_acc-cls_0.71695_mean-iu_0.60196_fwavacc_0.85163_lr_0.0003000000.pth'
    config.train_weights  = [1, 2,2,5]

elif config.exp_name.split('category')[-1] == '_stem':
    config.test_snapshot   = 'epoch_23_loss_5184.85547_acc_0.99633_acc-cls_0.99047_mean-iu_0.98280_fwavacc_0.99271_lr_0.0001000000.pth'
    config.test_hist_fname = 'Hist_category_stem.csv'
    config.test_csv_fname  = 'Portion_category_stem.csv'
    config.test_save_cls   = ["2=red", "3-brown"]

    config.train_snapshot = 'epoch_23_loss_5184.85547_acc_0.99633_acc-cls_0.99047_mean-iu_0.98280_fwavacc_0.99271_lr_0.0001000000.pth'
    config.train_weights  = [1, 1]


#
config.save_dir = os.path.join('../../result/', config.exp_name)
os.makedirs(config.save_dir, exist_ok=True)

config.test_hist_fname = os.path.join(config.save_dir, config.test_hist_fname)
config.test_csv_fname = os.path.join(config.save_dir, config.test_csv_fname)
config.test_analysis_mode = ['portion', 'hist', 'mean']


#
