import datetime
import os

import cv2
import numpy as np
from PIL import Image
import skimage.measure as smeasure

import torch
import torchvision.transforms as standard_transforms
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

import sys
sys.path.append('../../')
from datasets import plant
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
        net = FCN8s(num_classes=plant.num_classes).cuda()

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
        test_loader = DataLoader(test_set, batch_size=1, num_workers=1, shuffle=False)
        # with plant.gpu_mem_restore_ctx():
        for vi, data in enumerate(test_loader):
            img, img_info = data
            save_subpath  = img_info['sub_path']
            img_name      = img_info['img_name'][0]

            save_dir = os.path.join(cfg.save_dir, save_subpath[0])
            check_mkdir(save_dir)

            img      = Variable(img).cuda()
            output   = net(img)
            prediction   = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

            # do analysis and write to file
            rgbI      = plant.readRGBImage(save_subpath[0], img_name)
            tmp = prediction + 0

            if 'stem' in cfg.exp_name: # stem binary classification process
                prediction = tmp.astype(np.uint8)
                labelI = smeasure.label(cv2.morphologyEx(prediction, cv2.MORPH_OPEN, np.ones([7,7])))
                props = smeasure.regionprops(labelI)
                if len(props) > 1:
                    for prop in props:
                        cir_to_sqr = 4 * prop.area/((prop.bbox[2]-prop.bbox[0])*(prop.bbox[3]-prop.bbox[1]))
                        if cir_to_sqr < np.pi*0.98 or cir_to_sqr > np.pi*1.02:
                            labelI[labelI==prop.label] = 0
                    if np.sum(labelI) > 0:
                        prediction[labelI==0] = 0

                    if False and len(props) > 1: # get rid of not good stem prediction.
                        cir_to_sqr = 4 * props[0].area/((props[0].bbox[2]-props[0].bbox[0])*(props[0].bbox[3]-props[0].bbox[1]))
                        if cir_to_sqr < np.pi*0.98 or cir_to_sqr > np.pi*1.02:
                            continue
                # save prediction result
                save_img = Image.new('RGB', (2*rgbI.shape[1], rgbI.shape[0]))
                save_img.paste(Image.fromarray(rgbI), (0,0))
                pred_pil = plant.colorize_mask(prediction)
                save_img.paste(pred_pil, (rgbI.shape[1], 0))
                save_img.save(os.path.join(save_dir, img_name+'_cmp.jpg'))

                save_labelI = Image.new('P',(rgbI.shape[1], rgbI.shape[0]))
                save_labelI.putpalette(plant.palette)
                save_labelI.paste(Image.fromarray(prediction), (0,0))
                save_labelI.save(os.path.join(save_dir, img_name+'.png'))
            else:
                # get the crop predictions into the original size:
                prediction = np.zeros((img_info['ori_size'][1], img_info['ori_size'][0]), dtype=np.uint8)
                x0,y0,x1,y1 = img_info['crop_box']
                prediction[y0:y1, x0:x1] = tmp.astype(np.uint8)

                # save prediction result as image
                pred_pil   = plant.colorize_mask(prediction)
                save_img   = Image.new('RGB', (2*rgbI.shape[1], rgbI.shape[0]))
                save_img.paste(Image.fromarray(rgbI), (0,0))
                save_img.paste(pred_pil, (rgbI.shape[1], 0))
                save_img.save(os.path.join(save_dir, img_name + '_cmp.png'))

                save_labelI = Image.new('P',(rgbI.shape[1], rgbI.shape[0]))
                save_labelI.putpalette(plant.palette)
                save_labelI.paste(Image.fromarray(prediction), (0,0))
                save_labelI.save(os.path.join(save_dir, img_name+'.png'))

            if vi%1 == 0:
                print('%d / %d, %s' % (vi + 1, len(test_loader), img_name))


if __name__ == '__main__':
    main(args)
