import datetime
import os

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as standard_transforms
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

import sys
sys.path.append('../../')
from datasets import plant
from datasets.ds_utils import resize_map as resize_map
from models import *
from utils import check_mkdir
import utils.transforms as extended_transforms

from config import config as cfg


cudnn.benchmark = True

ckpt_path = '../../ckpt'

args = {
    'exp_name':cfg.exp_name,
    # empty string denotes learning from scratch
    'snapshot':cfg.test_snapshot
}

'''
def store_one_file(save_pickle, predI, img_name, dict_key='pred'):
        save_pickle[img_name] = {dict_key:predI}

def dump_segmentation_result(save_pickle, save_fname):
    with open(save_fname, 'wb') as f:
        pickle.dump(save_pickle, f, pickle.HIGHEST_PROTOCOL)
'''

def main(test_args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with torch.no_grad():
        net = PSPNet(num_classes=plant.num_classes).cuda()

        print('loading model from ' + test_args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, test_args['exp_name'], test_args['snapshot'])))
        net.eval()

        mean_std = ([0.385, 0.431, 0.452], [0.289, 0.294, 0.285])

        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])

        restore_transform = standard_transforms.Compose([
            extended_transforms.DeNormalize(*mean_std),
            standard_transforms.ToPILImage(),
        ])

        test_set    = plant.Plant('test', transform=input_transform)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)

        for vi, data in enumerate(test_loader):
            img, img_info = data
            save_subpath  = img_info['sub_path']
            img_name      = img_info['img_name'][0]
            rfn_stemI     = img_info['rfn_stemI'].cpu().numpy().squeeze()

            save_dir = os.path.join(cfg.save_dir, test_args['exp_name'], save_subpath[0])
            check_mkdir(save_dir)

            img          = Variable(img, volatile=True).cuda()
            output       = net(img)
            prediction   = np.transpose(output.cpu().numpy().squeeze(), [1,2,0])

            # get the crop predictions into the original size:
            x0,y0,x1,y1  = img_info['crop_box']
            tmp          = np.argmax(resize_map(prediction, (y1-y0, x1-x0)), axis=-1).astype(np.uint8)

            tmp[rfn_stemI==0]        = 0
            prediction               = np.zeros((img_info['ori_size'][1], img_info['ori_size'][0]), dtype=np.uint8)
            prediction[y0:y1, x0:x1] = tmp

            # do analysis and write to file
            rgbI      = plant.readRGBImage(save_subpath[0], img_name)

            # save prediction result as image
            pred_pil   = plant.colorize_mask(prediction[y0:y1, x0:x1])
            stem_pil   = plant.colorize_mask(rfn_stemI)
            save_img   = Image.new('RGB', (3*(x1-x0)+6, (y1-y0)))
            save_img.paste(Image.fromarray(rgbI[y0:y1, x0:x1,:]), (0, 0))
            save_img.paste(pred_pil, (x1-x0+3, 0))
            save_img.paste(stem_pil, (2*(x1-x0)+3, 0))
            save_img.save(os.path.join(save_dir, img_name + '_cmp.png'))

            save_labelI = Image.new('P',(rgbI.shape[1], rgbI.shape[0]))
            save_labelI.putpalette(plant.palette)
            save_labelI.paste(Image.fromarray(prediction), (0,0))
            save_labelI.save(os.path.join(save_dir, img_name+'.png'))

            if vi%1 == 0:
                print('%d / %d, %s' % (vi + 1, len(test_loader), img_name))

if __name__ == '__main__':
    main(args)
