import argparse
import scipy
import os
import cv2
import numpy as np
import json
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image

import dataloaders
import models
from utils.helpers import colorize_mask
from dataloaders.plant import get_image_list, parse_filename, makedir
from save_tool import SaveTool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import pdb

from pydensecrf.densecrf_refine import denseCRF_refine_one_image

def save_images(image, mask, output_path, image_file, palette, ext='.png'):
	# Saves the image, the model output and the results after the post processing
    w, h = image.size
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file+ext))
    # output_im = Image.new('RGB', (w*2, h))
    # output_im.paste(image, (0,0))
    # output_im.paste(colorized_mask, (w,0))
    # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))

def main():
    args = parse_arguments()
    config = json.load(open(args.config))
    imgSaver = SaveTool(margin=3)

    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K', 'PLANT']
    if args.mode == 'multiscale':
        if dataset_type == 'CityScapes':
            scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        else:
            scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    checkpoint = torch.load(args.model)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    rs_size = config['val_loader']['args']['crop_size']
    image_files = get_image_list(args.images, args.extension)
    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        for img_file in tbar:
            # read data
            sub_dir, img_name = parse_filename(img_file, args.extension)
            img_path = os.path.join(args.images, sub_dir, img_name+args.extension)
            rgb_img = cv2.imread(img_path)[..., [2,1,0]].astype(np.float32)

            # resize
            h, w = rgb_img.shape[:2]
            if h < w:
                h, w = (rs_size, int(rs_size * w / h))
            else:
                h, w = (int(rs_size * h / w), rs_size)
            rgb_img = cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray(rgb_img)

            # network
            input = normalize(to_tensor(image)).unsqueeze(0)
            pred_logit = model(input.to(device))
            pred_logit = pred_logit.squeeze(0)
            pred_prob  = F.softmax(pred_logit, dim=0).cpu().numpy()
            pred_prob  = np.transpose(pred_prob, [1,2,0])
            prediction = pred_prob.argmax(axis=-1)

            # save result
            result_path = os.path.join(args.output, sub_dir)
            makedir(result_path)
            vis_images = [rgb_img, prediction, pred_prob[..., 1] > 5e-2, pred_prob[..., 1]>1e-1]
            vis_palettes = ['RGB', 'label', 'label', 'label']
            texts = ['RGB', 'ArgMax', 'cls_1_0.05', 'cls_1_0.1']
            imgSaver.save_group_pilImage_RGB(vis_images,vis_palettes,texts, nr=1,nc=4,
                            save_path=os.path.join(result_path, img_name+'_pred.png'))

            norm_prob  = (pred_prob * 255.).astype(np.uint8)
            norm_prob  = norm_prob.transpose([0, 2, 1]).reshape([norm_prob.shape[0],-1])
            imgSaver.save_single_pilImage_gray(norm_prob, 'range',autoScale=False,
                            save_path=os.path.join(result_path, img_name+'_prob.png'))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='VOC',type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='singlescale', type=str,
                        help='Mode used for prediction: either [multiscale, singlescale]')
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='.jpg', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
