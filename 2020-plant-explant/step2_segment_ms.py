"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
    -- scipy.misc could read label image for scipy <= 1.20: scipy.misc.imread(fname, mode='P')
    -- new tool for label image reading is: skimage.io.imread(fname, pilmode='P')
"""

from __future__ import print_function
import os
import cv2
import numpy as np
import scipy.misc as smisc
from skimage import measure as smeasure
from PIL import Image

from config import get_image_list, parse_filename, makedir
from config import cfg
from save_tool import SaveTool
from pymeanshift import segment as mf_segment

from matplotlib import pyplot as plt
import pdb


def isolated_segment_based_color(rgb_img, bg_sureI):
    '''
    @func: segment out salient foreground based on color
    @param: rgb_img -- image in [RGB] mode
            bg_sureI -- binary image indicating background pixels
    '''
    rgb_img  = rgb_img.astype(np.float32)
    ht, wd   = rgb_img.shape[:2]

    # set rgb as 0 on background pixels out of petri-dish
    y, x = np.where(bg_sureI==0)
    y0, y1 = max(y.min()-15, 0), min(y.max()+15, ht-1)
    x0, x1 = max(x.min()-15, 0), min(x.max()+15, wd-1)
    fg_flagI = np.zeros_like(bg_sureI)
    fg_flagI[y0:y1, x0:x1] = 1

    # meanshift segment
    mf_spatial_radius = 11
    mf_range_radius   = 5
    mf_min_density    = 10
    _, mf_labelI, _   = mf_segment((rgb_img*fg_flagI[..., None]).astype(np.uint8),
                                   mf_spatial_radius, mf_range_radius, mf_min_density)
    mf_labelI = mf_labelI + 1

    return mf_labelI


def refine_segment(sem_prob, color_segI, rgbI):
    '''
    @func: refining the sem2nd_ret['sem'] w.r.t. color_segmentI
    @params:
           sem_prob -- array in [ht, wd, ch], probability map
           color_segI -- color based oversegment,
           rgbI -- 'RGB' image in [ht, wd, 3]
    '''
    ht, wd     = sem_prob.shape[:2]
    new_labelI = np.zeros([ht, wd])

    semI  = sem_prob.argmax(axis=-1)
    props = smeasure.regionprops(color_segI)
    for prop in props:
        coord  = prop.coords
        cls_ids, cnts = np.unique(semI[coord[:,0], coord[:,1]], return_counts=True)
        cls_id = cls_ids[cnts.argmax()]

        # decide class
        rgb_mean = rgbI[coord[:,0], coord[:,1], :].mean(axis=0)
        if (rgb_mean.max()<100) and (rgb_mean[1] < rgb_mean[2]+20):
            new_labelI[coord[:,0], coord[:,1]] = 0
        else:
            new_labelI[coord[:,0], coord[:,1]] = cls_id

    return new_labelI


def read_parse_network_segment(img_name, file_dir, file_ext, num_classes=4):
    I = np.asarray(Image.open(os.path.join(file_dir, img_name+file_ext)))
    ht, wd = I.shape
    probM = I.reshape([ht, num_classes, wd//num_classes]).transpose([0, 2, 1])
    probM = probM.astype(np.float32)/255.

    return probM


def main(option, imgSaver, RUN_COLOR_SEG=False, RUN_REFINE=False):
    # image to be process.
    imageList = get_image_list(option.rgb_dir, option.rgb_ext)

    for k, fname in enumerate(imageList):
        sub_dir, img_name = parse_filename(fname, option.rgb_ext)

        if False and \
           not (any(ele in img_name for ele in ['190313'])):
            #not (any(ele in sub_dir for ele in ['ETC_wk7'])):
            continue

        print("img {:d} | {:d}, {}".format(k, len(imageList), sub_dir+'/'+img_name))
        result_path = os.path.join(option.save_dir, sub_dir)
        makedir(result_path)
        temp_path = os.path.join(option.temp_dir, sub_dir)
        makedir(temp_path)

        # read rgb image and sem_1st
        rgb_path = os.path.join(option.rgb_dir, sub_dir, img_name+option.rgb_ext)
        rgb_img  = cv2.imread(rgb_path)[..., [2,1,0]]
        sem_prob =  read_parse_network_segment(img_name,
                                              os.path.join(option.sem_dir, sub_dir),
                                              option.sem_ext,
                                              num_classes=option.num_classes)
        ht, wd = sem_prob.shape[:2]
        rgb_img = cv2.resize(rgb_img, (wd, ht))

        # color based segment
        if RUN_COLOR_SEG:
            color_segI = isolated_segment_based_color(rgb_img, sem_prob[..., 0]>0.9)
            imgSaver.save_single_pilImage_gray(color_segI, 'label',
                        save_path=os.path.join(temp_path, img_name+'_clrSeg.png'))

        elif RUN_REFINE: # bypass sure-part to save time
            clrseg_path = os.path.join(temp_path, img_name+'_clrSeg.png')
            color_segI  = smisc.imread(clrseg_path, mode='P')
            color_segI  = smeasure.label(color_segI)

        # refine network segment using color segment
        if RUN_REFINE:
            # assign bg on color-seg
            sure_bgI = sem_prob[...,0]>0.9
            props = smeasure.regionprops(color_segI)
            for prop in props:
                coord = prop.coords
                bg_cnt = sure_bgI[coord[:,0], coord[:,1]].sum()
                if bg_cnt > prop.area*0.5:
                    color_segI[coord[:,0], coord[:,1]] = 0
            color_segI = smeasure.label(color_segI)

            # refine
            final_semI = refine_segment(sem_prob, color_segI, rgb_img)

            # save result
            vis_images = [rgb_img, sem_prob.argmax(axis=-1), color_segI, final_semI]
            vis_palettes = ['RGB', 'label', 'label', 'label']
            vis_texts = ['RGB', 'cnn seg', 'color seg', 'rfn seg']
            save_path=os.path.join(result_path, img_name+'_grid.png')
            imgSaver.save_group_pilImage_RGB(vis_images,vis_palettes, vis_texts,
                                             nr=2,nc=2,resize=256, save_path = save_path)

            save_path=os.path.join(result_path, img_name+option.save_ext)
            imgSaver.save_single_pilImage_gray(final_semI, 'label', save_path=save_path)


if __name__ == '__main__':
    imgSaver = SaveTool()

    main(cfg.segment,
         imgSaver,
         RUN_COLOR_SEG=True,
         RUN_REFINE=True)


