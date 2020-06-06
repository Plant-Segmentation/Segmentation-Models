from easydict import EasyDict as edict

import os
import os.path as osp


config = edict()

# semantic segmentaiton config 
config.label_path   = 'category4'

config.num_classes  = 4

config.ignore_label = 255

config.class_name = ['background', 'stem', 'callus', 'shoot']

config.weights = [1, 2.2, 2.5, 4.4] #background, stem, callus, shoot


