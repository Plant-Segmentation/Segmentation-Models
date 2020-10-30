import os.path as osp

import numpy as np
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

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'test':
        img_path = osp.join(cfg.root_path, 'images')
        data_list = [l.strip('\n') for l in open(osp.join(
            cfg.root_path, 'AnnotateResult', 'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path, it))
    else:
        if mode == 'train':
            img_path = osp.join(cfg.root_path, 'images')
            mask_path = osp.join(cfg.root_path, 'AnnotateResult', cfg.label_path, 'labels')
            data_list = [l.strip('\n') for l in open(osp.join(
                cfg.root_path, 'AnnotateResult',cfg.label_path, 'train.txt')).readlines()]
        else:
            assert(mode == 'val')
            img_path = osp.join(cfg.root_path, 'images')
            mask_path = osp.join(cfg.root_path, 'AnnotateResult', cfg.label_path, 'labels')
            data_list = [l.strip('\n') for l in open(osp.join(
                cfg.root_path, 'AnnotateResult', cfg.label_path,'val.txt')).readlines()]

        for it in data_list:
            it_path, it_name = it.split('/')
            if 'high' in it_path:
                rgb_path = osp.join(img_path, cfg.trainval_rgb_path['high'], it_name+'.jpg')
            else:
                rgb_path = osp.join(img_path, cfg.trainval_rgb_path['others'], it_name+'.jpg')
            sem_path = osp.join(mask_path, it_path, it_name+'.png')
            items.append((rgb_path, sem_path))

    return items

def readRGBImage(sub_path, fname):
    img = Image.open(osp.join(cfg.root_path, 'images', sub_path, fname+'.jpg')).convert('RGB')
    return np.asarray(img)

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
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(osp.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            #return cfg.img_save_dict[img_name.split('/')[0]], img_name.split('/')[1], img
            return ('/').join(img_name.split('/')[:-1]), img_name.split('/')[-1], img

        img_path, mask_path = self.imgs[index]
        if self.mode == 'val':
            img  = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path)
        else: # 'train'
            img  = smisc.imread(img_path)
            mask = smisc.imread(mask_path, mode='P')
            if self.augmentations is not None:
                img, mask = self.augmentations(img, mask)
            #img  = cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)
            #mask = cv2.resize(mask, (500, 500), interpolation=cv2.INTER_NEAREST)
            img  = Image.fromarray(img.astype(np.uint8))
            mask = Image.fromarray(mask.astype(np.uint8))

        img  = img.resize((500,500), Image.BILINEAR)
        mask = mask.resize((500,500), Image.NEAREST)

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
