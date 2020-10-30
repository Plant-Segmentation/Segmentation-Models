import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import yaml
import cv2
import argparse
import scipy.io as scpio
from PIL import Image
from skimage.transform import resize
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from scipy import ndimage
import math

class_color = [[0,   0, 0], # 0-background
                         [128, 0, 0], # 1-stem
                         [0, 0, 128], # 2-callus
                         [0, 128, 0], # 3-shoot
                          ]
idx_palette = np.reshape(np.asarray(class_color), (-1))

def check_point_within(points, image):
    '''
        Return even if one point belongs to the ellipse.
    '''
    center = (500, 500)
    a, b = 445, 600
    for pt in points:
        dist = ((math.pow((center[0]-pt[0]), 2) / math.pow(a, 2)) + (math.pow((center[1]-pt[1]), 2) / math.pow(b, 2)))
        if dist<=1:
            return True
    return False

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

    def __init__(self, model_path):
        """Creates and loads pretrained deeplab model."""
        frozen_graph_filename = model_path+self.FROZEN_GRAPH_NAME
        self.graph = tf.Graph()

        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image, INPUT_SIZE = 900):
        """Runs inference on a single image.
        Args:
        image: A PIL.Image object, raw input image.

        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size      
        resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})      
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_invitro_label_colormap():
    """
    Creates a colormap for the invitro segmentation.

    Returns: 
    A colormap for visualizing the results.
    """
    colormap = np.zeros((4, 3), dtype=np.uint8)
    colormap[0]=[0, 0, 0] # background (class 0)
    colormap[1]=[128, 0, 0] # stem (class 1)
    colormap[2]=[0, 0, 128] # callus (class 2)
    colormap[3]=[0, 128, 0] # shoot (class 3)
    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
    is the color indexed by the corresponding element in the input label
    to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
    map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_invitro_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def main():
    download_cfg = yaml.safe_load(open('./deeplab/semantic_config.yaml'))

    model_path = download_cfg['DATASET']['MODEL_DIR']
    BASE_PATH = download_cfg['SEMANTIC SEG']['BASE_PATH']
    RGB_PATH = download_cfg['SEMANTIC SEG']['RGB_PATH']
    image_path = os.path.join(BASE_PATH, RGB_PATH)
    IMAGE_LISTS = download_cfg['SEMANTIC SEG']['TEST_LIST']
    SEG_PATH = download_cfg['SEMANTIC SEG']['SEG_RESULT']
    segment_path = os.path.join(BASE_PATH, SEG_PATH)
    if not os.path.exists(segment_path):
        os.makedirs(segment_path)

    MODEL = DeepLabModel(model_path)

    # Read the training list and load the image names in the lists
    f = open(IMAGE_LISTS,'r')
    image_list = []

    for line in f:
        img_name = line.strip("\n")
        image_list.append(img_name)
    image_list = list(set(image_list))
    print("Running inference for ", len(image_list), " images.")

    # run semantic segmentation on each image and save it.    
    for e, img_name in enumerate(image_list):   
        print("Running deeplab on image: "+ str(e+1))    
        output_path = os.path.join(segment_path, img_name+download_cfg['SEMANTIC SEG']['SEG_EXT']) 
        image = Image.open(os.path.join(image_path, img_name+download_cfg['SEMANTIC SEG']['RGB_EXT']))      
        width, height = image.size
        _, seg_map = MODEL.run(image, max(width, height))  

        assert seg_map.shape[0]==height and seg_map.shape[1]==width

        # segment_res=Image.fromarray(seg_map).resize((image.size[0], image.size[1]), Image.NEAREST)
        # resized_image = cv2.resize(seg_map, (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_path, seg_map)
        
        seg_map = np.array(Image.fromarray(seg_map).resize((1000, 1000)))
        gray = cv2.cvtColor(cv2.imread(output_path), cv2.COLOR_BGR2GRAY)
        gray = np.array(cv2.resize(gray, (1000, 1000)))
        gray[gray!=0] = 1

        # find the connected components with statistics
        output = cv2.connectedComponentsWithStats(gray, 4)
        num_labels = output[0]
        labels = output[1]

        for lbl in range(num_labels):
            centroid = output[3][lbl].reshape(1, 2)
            within = check_point_within(centroid, gray)
            
            if not within: 
                points = list(zip(*np.where(labels==lbl)))
                
                # check for all the points belonging to that label.
                within = check_point_within(points, seg_map)
                if not within:
                    seg_map[labels==lbl] = 0

        seg_map = Image.fromarray(seg_map).resize((width, height))

        save_labelI = Image.new('P', (image.size[0], image.size[1]))
        save_labelI.putpalette(list(idx_palette))
        save_labelI.paste(seg_map, (0,0))
        save_labelI.save(output_path)

if __name__ == '__main__':
    main()  
