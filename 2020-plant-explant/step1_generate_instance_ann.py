'''
Generating instance ann from semantic ann. Using the isolation prior knowledge:
    CUDA_VISIBLE_DEVICES=0 python train.py --config config/config_explant_20.json

    CUDA_VISIBLE_DEVICES=1 python infer_explant.py --config config/config_explant_12.json --model saved/PSPNet/PLANT_Explant_grid12/04-04_15-55/checkpoint-epoch150.pth --images /media/yuanjial/LargeDrive/DataSet/Forest-Explant/grid-12/images/ --output /media/yuanjial/LargeDrive/DataSet/Forest-Explant/grid-12/step1_semantic/

'''

import os
import numpy as np
import scipy.misc as smisc
from skimage import measure as smeasure
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from config import get_image_list, parse_filename, makedir
from config import cfg
from save_tool import SaveTool

import pdb


def save_colorbar(imgSaver, save_path='./color_bar.png'):
    images, texts = [], []
    for k in range(1, 21):
        img = np.zeros([32,32], dtype=np.uint8) + k
        images.append(img)
        texts.append(str(k))

    palettes = ['Label']*len(texts)
    imgSaver.save_group_pilImage_RGB(images, palettes, texts,
                                     nr=5, nc =4, fontsize=10,
                                     save_path=save_path)

def main(option):
    imgSaver = SaveTool()
    #save_colorbar(imgSaver)

    pdb.set_trace()
    imageList = get_image_list(option.sem_dir, option.sem_ext)
    for k, fname in enumerate(imageList):
        sub_dir, img_name = parse_filename(fname, option.sem_ext)
        print("img {:d} | {:d}, {}".format(k, len(imageList), img_name))
        result_path = os.path.join(option.save_dir, sub_dir)
        makedir(result_path)

        # read image
        sem_path = os.path.join(option.sem_dir, sub_dir, img_name+option.sem_ext)
        semI     = smisc.imread(sem_path, mode='P')
        ht, wd   = semI.shape

        # generating proposalsremove noise
        props    = smeasure.regionprops(smeasure.label(semI>0))
        areas    = [prop.area for prop in props]
        sort_idx = sorted(range(len(props)), key=lambda i: -areas[i])
        props    = [props[i] for i in sort_idx[:option.num_grid]]

        if len(props) < len(option.col_ratio):
            continue
        else:
            labelI  = np.zeros_like(semI)
            # separate based on rows.
            cen_Ys  = np.asarray([prop.centroid[0] for prop in props])[:, None]
            kmeans_func0 = KMeans(n_clusters=len(option.row_ratio),
                                  n_init = 1,
                                  init=ht*option.row_ratio[:,None])
            row_labels = kmeans_func0.fit(cen_Ys).labels_

            # separate based on cols.
            cen_Xs   = np.asarray([prop.centroid[1] for prop in props])[:, None]
            kmeans_func1 = KMeans(n_clusters=len(option.col_ratio),
                                  n_init = 1,
                                  init=wd*option.col_ratio[:,None])
            col_labels = kmeans_func1.fit(cen_Xs).labels_

            # assigning label
            for i in range(len(props)):
                coord    = props[i].coords
                row, col = row_labels[i], col_labels[i]
                label_id = option.row_size*row + col + 1
                if str(label_id) in option.lut:
                    labelI[coord[:,0], coord[:,1]] = option.lut[str(label_id)]

        save_path=os.path.join(result_path, img_name+option.save_ext)
        imgSaver.save_single_pilImage_gray(labelI, 'label', save_path=save_path)


if __name__ == "__main__":
    main(cfg.isolate)
