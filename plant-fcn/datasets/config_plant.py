from easydict import EasyDict as edict

import os
import os.path as osp
#pwd_path = os.getcwd()

config = edict()

# pathes
config.root_path = '/home/yuanjial/Projects/pyTorch/pytorch-semantic-segmentation/datasets/Plant-data/'
config.source_image_crop = True

# -----------------------------------------
# semantic segmentaiton config
if False: # 11 categories
    config.label_path   = 'category11'
    config.num_classes  = 11
    config.ignore_label = 255
    config.class_name = ['background', 'stem', 'callus', 'callus red', 'shoot',\
                            'shoot red', 'leaf', 'shoot brown', 'callus brown', \
                            'leaf brown', 'root']
    config.save_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [127,0,128], # 3-callus red
                         [0, 128, 0], # 4-shoot
                         [0,128,128], # 5-shoot red/red1
                         [128,128,0], # 6-leaf
                         [64, 0,128], # 7-shoot brown
                         [64, 0,  0], # 8-callus brown
                         [0,  0, 64], # 9-leaf brown
                         [0, 64,  0], # 10-root
                          ]

elif False: # 8 categories
    config.label_path   = 'category8'
    config.num_classes  = 8
    config.ignore_label = 255

    config.class_name = ['background', 'stem', 'callus', 'callus red', 'callus brown',\
                                               'shoot', 'shoot red', 'shoot brown']
    config.save_color = [[  0,   0,   0], # 0-background
                         [128,   0,   0], # 1-stem
                         [  0,   0, 128], # 2-callus
                         [127,   0, 128], # 3-callus red
                         [  0, 128,   0], # 4-callus brown
                         [  0, 128, 128], # 5-shoot
                         [128, 128,   0], # 7-shoot red
                         [ 64,  64,   0], # 8-shoot brown
                          ]

elif False: # 7 categories
    config.label_path   = 'category7'
    config.num_classes  = 7
    config.ignore_label = 255
    config.class_name = ['background', 'stem', 'callus', 'callus red', 'shoot',\
                            'shoot red', 'leaf']
    config.save_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [127,0,128], # 3-callus red
                         [0, 128, 0], # 4-shoot
                         [0,128,128], # 5-shoot red/red1
                         [127,128,0], # 6-leaf
                          ]

elif False: # 5 categories.
    config.label_path   = 'category5'
    config.num_classes  = 5
    config.ignore_label = 255
    config.class_name = ['background', 'stem', 'callus', 'shoot', 'leaf']
    config.save_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [0, 128, 0], # 3-shoot
                         [127,0,128], # 4-leaf
                          ]

elif True: # 4 categories.
    config.label_path   = 'category4'
    config.num_classes  = 4
    config.ignore_label = 255
    config.class_name = ['background', 'stem', 'callus', 'shoot']
    config.save_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [0, 128, 0], # 3-shoot
                          ]

elif False: # 4 categories about color
    config.label_path = 'category_color'
    config.num_classes  = 4
    config.ignore_label = 255
    config.class_name = ['background', 'green', 'red', 'brown']
    config.save_color = [[  0,   0, 0], # 0-background
                         [  0, 128, 0], # 1-green
                         [128,   0, 0], # 2-red
                         [204, 136, 0], # 3-brown
                          ]

elif False: # 2 categories about stem / non stem
    config.source_image_crop = False
    config.label_path   = 'category_stem'
    config.num_classes  = 2
    config.ignore_label = 255
    config.class_name = ['background', 'stem']
    config.save_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                          ]
# ---------------------------------------------------------
