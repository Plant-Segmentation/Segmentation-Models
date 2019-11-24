import os.path as osp

import numpy as np
from matplotlib import pyplot as plt
import scipy.misc as smisc
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

import cv2
from datasets.config_plant import config as cfg


num_classes  = cfg.num_classes
ignore_label = cfg.ignore_label
experiment   = cfg.label_path

# palette for saving predicting result.
palette = list(np.reshape(np.asarray(cfg.save_color), (-1)))
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

range_palette = np.reshape(np.floor(np.asarray(plt.get_cmap('viridis').colors)*256), [-1]).astype(np.uint8)

def colorize_mask(mask, palette_type='quantized'):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    if palette_type=='quantized':
        new_mask.putpalette(palette)
    else: # 'range'
        new_mask.putpalette(range_palette)

    return new_mask


def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    img_path = osp.join(cfg.root_path, 'images')

    items = []
    if mode == 'test':
        data_list = [l.strip('\n') for l in open(osp.join(
            cfg.root_path, 'AnnotationResult', 'test.txt')).readlines()]
        for it in data_list:
            rgb_path = osp.join(img_path, it+'.jpg')
            rfn_path = osp.join(img_path, 'binary_detect', it+'.png') # images for crop foreground
            items.append((rgb_path, rfn_path, it))
    else:
        mask_path = osp.join(cfg.root_path, 'AnnotationResult', cfg.label_path, 'labels')
        if mode == 'train':
            data_list = [l.strip('\n') for l in open(osp.join(
                cfg.root_path, 'AnnotationResult',cfg.label_path, 'train.txt')).readlines()]
        else:
            assert(mode == 'val')
            data_list = [l.strip('\n') for l in open(osp.join(
                cfg.root_path, 'AnnotationResult', cfg.label_path,'val.txt')).readlines()]

        for it in data_list:
            it_path, it_name = it.split('/')
            rgb_path = osp.join(img_path, 'trainval_images', it_name+'.jpg')
            rfn_path = osp.join(img_path, 'binary_detect/trainval_images', it_name+'.png')  # images for crop foreground
            sem_path = osp.join(mask_path, it_path, it_name+'.png')
            items.append((rgb_path, sem_path, rfn_path))

    return items

def readRGBImage(sub_path, fname):
    img = Image.open(osp.join(cfg.root_path, 'images', sub_path, fname+'.jpg')).convert('RGB')
    return np.asarray(img)

def cropboxFromRefineImage(rfn_path):
    rfn_img = smisc.imread(rfn_path, mode='P')

    ht, wd = rfn_img.shape
    y, x = np.where(rfn_img>0)
    if len(x) == 0:
        x0, y0, x1, y1 = 0, 0, wd, ht
    else:
        ext_ht = 256
        ext_wd = (ext_ht*wd)//ht
        x0, y0 = max(0, min(x)-ext_wd), max(0, min(y)-ext_ht)
        x1, y1 = min(wd, max(x)+ext_wd), min(ht, max(y)+ext_ht)

    return (x0,y0,x1,y1), rfn_img[y0:y1, x0:x1]


class Plant(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, augmentations=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop    = sliding_crop
        self.augmentations   = augmentations
        self.transform       = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # test mode
        if self.mode == 'test':
            img_path, rfn_path, sub_name = self.imgs[index]
            img      = Image.open(img_path).convert('RGB')
            if cfg.source_image_crop == True: # focus to center if needed
                ori_size            = img.size  # (wd, ht)
                rfn_box, rfn_stemI  = cropboxFromRefineImage(rfn_path)  # (x0,y0,x1,y1)
                img                 = img.crop(rfn_box)
            else:
                ori_size, rfn_box, rfn_stemI = 'None', 'None', 'None'

            img = img.resize((500, 500), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)

            img_info = {'sub_path':('/').join(sub_name.split('/')[:-1]),
                        'img_name':sub_name.split('/')[-1],
                        'crop_box':rfn_box,
                        'ori_size':ori_size,
                        'rfn_stemI': rfn_stemI}
            return img, img_info

        # train / val mode.
        img_path, mask_path, rfn_path = self.imgs[index]
        img  = smisc.imread(img_path)
        mask = smisc.imread(mask_path, mode='P')
        if cfg.source_image_crop == True: # focus to center if needed
            x0,y0,x1,y1  = cropboxFromRefineImage(rfn_path)  # (x0,y0,x1,y1)
            img          = img[y0:y1, x0:x1,:]
            mask         = mask[y0:y1,x0:x1]

        if self.mode == 'train' and self.augmentations is not None:
            img, mask = self.augmentations(img, mask)

        img  = Image.fromarray(img.astype(np.uint8)).resize((500, 500), Image.BILINEAR)
        mask = Image.fromarray(mask.astype(np.uint8)).resize((500, 500), Image.NEAREST)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)
'''
import traceback
class gpu_mem_restore_ctx():
    "context manager to reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted"
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val: return True
        traceback.clear_frames(exc_tb)
        raise exc_type(exc_val).with_traceback(exc_tb) from None
'''
