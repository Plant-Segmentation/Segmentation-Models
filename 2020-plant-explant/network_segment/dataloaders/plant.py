# Originally written by Kazuto Nakashima
# https://github.com/kazuto1011/deeplab-pytorch

from glob import glob
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2

from base import BaseDataSet, BaseDataLoader
from utils import palette

from torch.utils.data import Dataset
#from torchvision import transforms

######################################################
def get_image_list(data_dir, data_ext):
    dir_list = os.listdir(data_dir)

    image_set_index = []
    for fdir in dir_list:
        glob_imgs = glob(os.path.join(data_dir, fdir, '*'+data_ext))

        # get the image name in sorted order
        tmp_imgs  = sorted([ele.split('_') for ele in glob_imgs])
        glob_imgs = ['_'.join(ele) for ele in tmp_imgs]

        img_list = [(fdir, os.path.basename(v)) for v in glob_imgs]
        image_set_index += img_list

    return image_set_index

def parse_filename(fname, ext='.jpg'):
    if isinstance(fname, tuple):
        sub_dir, img_name = fname
    else:
        sub_dir, img_name = '', fname
    img_name = img_name.split(ext)[0]
    return sub_dir, img_name

def makedir(path):
    # exist_ok only work for python3.2 +
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


######################################################


class PlantDataset(BaseDataSet):
    """
    Plant dataset
    """
    def __init__(self, num_classes, rgb_ext, ann_ext, **kwargs):
        self.rgb_ext = rgb_ext
        self.ann_ext = ann_ext
        self.num_classes = num_classes
        self.palette = palette.get_voc_palette(self.num_classes)
        super(PlantDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root)
        self.image_dir = self.root
        self.label_dir = self.root

        self.files = get_image_list(self.root, self.rgb_ext)


    def _load_data(self, index):
        fname = self.files[index]
        sub_dir, img_name = parse_filename(fname, self.rgb_ext)
        image_id = os.path.join(sub_dir, img_name)

        image_path = os.path.join(self.image_dir, image_id + self.rgb_ext)
        label_path = os.path.join(self.label_dir, image_id + self.ann_ext)

        image = cv2.imread(image_path)[..., [2,1,0]].astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)

        if(len(image.shape) != 3 or len(label.shape) != 2):
            print('\n-- ', fname, 'image ', image.shape,  'label', label.shape)

        return image, label, image_id


class PLANT(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, \
                       base_size=None, scale=True, num_workers=1, val=False,\
                       shuffle=False, flip=False, rotate=False, blur= False,\
                       num_classes=4, rgb_ext='.jpg', ann_ext='.png',\
                       augment=False, val_split= None, return_id=False):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = PlantDataset(num_classes,rgb_ext, ann_ext,  **kwargs)
        super(PLANT, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

