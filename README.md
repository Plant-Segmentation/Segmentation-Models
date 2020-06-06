# Segmentation-Models
This repository contains models for image semantic segmentation. These models were used for segmenting plant traits for analysis.

## 1. Deeplab model 
The deeplab implementation is from the [repo](https://github.com/tensorflow/models/tree/master/research/deeplab). 

To perform training using the deeplab model, follow the following steps -

### 1.1 Prepare dataset 

    Preprocess the Dataset to convert them into TFRecords
    a) Split the dataset into training and validation either in a 80:20 or 85:15 ratio depending on the size of your dataset.
    - train.txt
    - val.txt 

    Please note that the labels used for training using deeplab should be continuous values beginning from 0.

    b) generate TFRecords 
    Please refer to the script to generate tfrecord - https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/download_and_convert_voc2012.sh

    Use the commands below-

    cd datasets/

    run the command -
    python ./build_plant_data.py \
  --image_folder="./Plant-data/JPEG" \
  --semantic_segmentation_folder="./Plant-data/AnnotationResult/category4/labels_gray" \
  --list_folder="./Plant-data/AnnotationResult/category4/Segmentation-labels/" \
  --image_format="jpg" \
  --output_dir="./Plant-data/tfrecord"

    For example,
  ** No of training examples - 102 (80% of 127)
  ** No of trainingval examples - 25 (20% of 127)

### 1.2 Add the dataset information in the data_generator file.
    
    datasets/data_generator.py

    _PLANT_CATEGORY4_INFORMATION = DatasetDescriptor(
        splits_to_sizes={
            'train': 102,    # num of samples in images/training
            'trainval': 25,  # num of samples in images/validation
            'test': 127,
        },
        num_classes=4,
        ignore_label=255,
    )

    _DATASETS_INFORMATION = {
        'cityscapes': _CITYSCAPES_INFORMATION,
        'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
        'ade20k': _ADE20K_INFORMATION,
        'category4': _PLANT_CATEGORY4_INFORMATION,
    }

### 1.3 Modify the Colormap Configuration for the dataset.
     utils/get_dataset_colormap.py

    Note: Name of dataset == _CATEGORY4
    use this function - label_to_color_image(label, dataset=_CATEGORY4)

### 1.4 Modify utils/train_utils.py to assign weights for different classes. Please modify this information according to the number of classes in the dataset.
    not_ignore_mask = \
                tf.to_float(tf.equal(scaled_labels, 0))*cfg.weights[0] + \
                tf.to_float(tf.equal(scaled_labels, 1))*cfg.weights[1] + \
                tf.to_float(tf.equal(scaled_labels, 2))*cfg.weights[2] + \
                tf.to_float(tf.equal(scaled_labels, 3))*cfg.weights[3] + \
                tf.to_float(tf.equal(scaled_labels, ignore_label))*cfg.weights[4]

    Add the configuration in config/config_plant.py

### 1.5 In train.py
    set params in the common.py and train.py present in the deeplab folder.

    For example, 
    image_pyramid = [0.25, 0.5, 1]
    
    If you do not wish to load the weights of the last layer, set initialize_last_layer = False. 
    
    Set the path of the train logdir, train_logdir="checkpoints/"
   
    Modify other hyperparameters - (It is important to set these parameters according to your own dataset.)
    Example -
    base_learning_rate=1e-3
    learning_rate_decay_step=500
    learning_rate_decay_factor=0.1 (power)
    
### 1.6 Run the training script 
    Slim repository can be found in the original implementation of the deeplab found [here](https://github.com/tensorflow/models/tree/master/research/deeplab). Place the slim repository outside the deeplab folder.

    Run the following commands from one directory above the deeplab folder.

    a) Add the slim path, using -
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

    b) Run the training script using -
    python deeplab/train.py

    You can choose from different backbone network available for network training (ResNet, Xception, MobileNet). 

    The pretrained models for Deeplab can be found [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md).

    
## 2. Random Forest Implementation
    Image segmentation using random forest. 

    The original implementation of the code is from the [repo](https://github.com/dgriffiths3/ml_segmentation.git).

### 2.1 Install the required dependencies, using -
    pip install -r requirements.txt

### 2.2 Modify parameters in rf_segmentations.py

    Modify the value of hyperparameters in rf_segmentations.py
    
    Please note - Modifications maybe required to the code for multiprocessing based on the number of images in the training dataset. 

    Multiple feature maps extracted from our input RGB image are listed below -

    Color Features: 
    - Red, Green, Blue channels from the RGB color space. 
    - RGB -> HSI 
    - RGB -> LAB  
    Lightness, Red/Green Value, Blue/Yellow, Saturation (color intensity), Hue (angular color) values at each pixel from these color spaces.

    Texture Features: 
    - Local Binary Patterns are obtained by thresholding the value of each pixel based on it's neighborhood pixel values. 
    - 9 Haralick Texture features including Angular second moment, Contrast, Correlation, Sum of square: Variance, Sum Variance, Sum Entropy, Entropy, Inverse Different Moment, and sum average.

    Edge Detectors: 
    - Sobel Kernels are used to extract gradient components in both horizontal and vertical directions. 

### 2.2 Train and test random forest.
    python rf_segmentations.py --train_list 'path to the training list' --test_list 'Path to the testing list'
    
## 3. PSPNet
    This repository is adopted from [here](https://github.com/yassouali/pytorch_segmentation). Complete instructions to run this model can be found on the same link.

    This code was used for GWAS traits segmentations.

### 3.1 Modify the config.py file.
    Add configurations for your dataset in the configuration file config.py.
