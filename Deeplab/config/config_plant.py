from easydict import EasyDict as edict

import os
import os.path as osp
#pwd_path = os.getcwd()

config = edict()

# semantic segmentaiton config
if True: # 4 categories.
    config.label_path   = 'category4'
    config.num_classes  = 4
    config.ignore_label = 255
    config.class_name = ['background', 'stem', 'callus', 'shoot']
    config.weights = [1, 2.2, 2.5, 4.4, 0] #[1, 250, 273, 262, 0] #[1, 12, 20, 47, 0] #background, stem, callus, shoot, ignore_label

# ---------------------------------------------------------

