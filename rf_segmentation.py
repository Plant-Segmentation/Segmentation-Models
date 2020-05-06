'''
    Thanks to the wonderful implementation of random forest in Python.
    https://github.com/dgriffiths3/ml_segmentation.git

    Modified script to accomodate the following -
    1. parallel feature processing
    2. model training parameters
    3. resize test images and store the results.
    Features added - 

'''
import cv2
import numpy as np
import pylab as plt
from glob import glob
import argparse
import os
import progressbar
import pickle as pkl
from numpy.lib import stride_tricks
from skimage import feature
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import mahotas as mt
import plant
from sklearn.utils.class_weight import compute_class_weight
from skimage import filters
from scipy.ndimage import gaussian_filter
from skimage.io import imread, imsave
import multiprocessing as mp

print("No of processors: ", mp.cpu_count())

def check_args(args):

    if not os.path.exists(args.image_dir):
        raise ValueError("Image directory does not exist")

    if not os.path.exists(args.label_dir):
        raise ValueError("Label directory does not exist")

    if args.classifier != "SVM" and args.classifier != "RF" and args.classifier != "GBC":
        raise ValueError("Classifier must be either SVM, RF or GBC")

    if args.output_model.split('.')[-1] != "pkl":
        raise ValueError("Model extension must be .pkl")

    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir" , help="Path to images", default= './dataset/JPEGImages/GWAS-callus-shoot/trainval_images/', required=False)
    parser.add_argument("-l", "--label_dir", help="Path to labels", default='./dataset/AnnotationResult/GWAS-callus-shoot/category4/labels/', required=False)
    parser.add_argument("-c", "--classifier", help="Classification model to use", default='RF', required=False)
    parser.add_argument("-o", "--output_dir", help="Path to save the outputs.", default='./output_dir', required=False)
    parser.add_argument("-m", "--output_model", help="Output model filename.", default='model8.pkl', required=False)
    parser.add_argument("-r", "--test_resize", help="Resize testing set images.", action='store_false')
    args = parser.parse_args()
    return check_args(args)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def read_data(image_dir, label_dir):

    print ('[INFO] Reading image data.')

    filelist = open('./dataset/AnnotationResult/GWAS-callus-shoot/category4/train.txt').readlines()
    image_list = []
    label_list = []

    for file in filelist:
        file = file.strip()
        image_list.append(cv2.imread(os.path.join(image_dir, file+'.jpg'), 1))
        label = imread(os.path.join(label_dir, file+'.png'), pilmode = "P")
        label_list.append(label)

    return image_list, label_list

def read_test_data(image_dir, label_dir, resize=True):
    '''
        Creating test dataset.
    '''
    print ('[INFO] Reading image data.')

    filelist = open('./dataset/AnnotationResult/GWAS-callus-shoot/category4/val.txt').readlines()
    image_list = []
    label_list = []

    for file in filelist[:2]:
        file = file.strip()
        image = cv2.imread(os.path.join(image_dir, file+'.jpg'), 1)
        label = imread(os.path.join(label_dir, file+'.png'), pilmode = "P")
        
        if resize: 
            image = cv2.resize(image, (200, 200))
            label = cv2.resize(label, (200, 200))

        image_list.append(image)
        label_list.append(label)

    return image_list, label_list

def subsample(features, labels, low, high, sample_size):

    idx = np.random.randint(low, high, sample_size)

    return features[idx], labels[idx]

def subsample_idx(low, high, sample_size):

    return np.random.randint(low,high,sample_size)

def calc_haralick(roi):

    feature_vec = []

    texture_features = mt.features.haralick(roi)
    mean_ht = texture_features.mean(axis=0)

    [feature_vec.append(i) for i in mean_ht[0:9]]

    return np.array(feature_vec)

def harlick_features(img, h_neigh, ss_idx):

    print ('[INFO] Computing haralick features.')
    size = h_neigh
    shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
    strides = 2 * img.strides
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
    patches = patches.reshape(-1, size, size)

    if len(ss_idx) == 0 :
        bar = progressbar.ProgressBar(maxval=len(patches), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    else:
        bar = progressbar.ProgressBar(maxval=len(ss_idx), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()

    h_features = []

    if len(ss_idx) == 0:
        for i, p in enumerate(patches):
            bar.update(i+1)
            h_features.append(calc_haralick(p))
    else:
        for i, p in enumerate(patches[ss_idx]):
            bar.update(i+1)
            h_features.append(calc_haralick(p))

    #h_features = [calc_haralick(p) for p in patches[ss_idx]]
    print ('[INFO] Haralick features competed.')

    return np.array(h_features)

def create_binary_pattern(img, p, r):

    print ('[INFO] Computing local binary pattern features.')
    lbp = feature.local_binary_pattern(img, p, r)
    return (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp)) * 255

def create_features(i, return_dict_features, return_dict_labels, img, img_gray, img_lab, img_hsv, label, train=True, trainval=False):

    lbp_radius = 24 # local binary pattern neighbourhood
    h_neigh = 11 # haralick neighbourhood
    num_examples = 2000 # number of examples per image to use for training model

    lbp_points = lbp_radius*8
    h_ind = int((h_neigh - 1)/ 2)
    
    sobel_img_h = filters.sobel_h(img_gray)
    sobel_img_v = filters.sobel_v(img_gray)
    sobel_img = filters.sobel(img_gray)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
    lap_img = cv2.Laplacian(img, cv2.CV_64F)
    angle = np.zeros((img.shape[0], img.shape[1], 3))
    angle[grad_x!=0] = np.arctan(grad_y[grad_x!=0]/grad_x[grad_x!=0])
    angle[grad_x==0] = 90 

    feature_img = np.zeros((img.shape[0],img.shape[1], 23))
    feature_img[:,:,:3] = img
    feature_img[:,:,4:6] = img_hsv[:,:,1:3] # saturation and value channels from HSV color image.
    feature_img[:,:,6:9] = img_lab
    feature_img[:,:,9] = sobel_img_h # 56 did not have gradients
    feature_img[:,:,10] = sobel_img_v
    feature_img[:,:,11] = sobel_img
    feature_img[:,:,12] = gaussian_filter(img_gray, sigma=5)
    feature_img[:,:,13] = gaussian_filter(img_gray, sigma=3)
    feature_img[:,:,14:17] = lap_img
    feature_img[:,:,17:20] = np.sqrt(np.square(grad_x)+np.square(grad_y))
    feature_img[:,:,20:23] = angle

    img = None
    feature_img[:,:,3] = create_binary_pattern(img_gray, lbp_points, lbp_radius)
    if train==True: feature_img = feature_img[h_ind:-h_ind, h_ind:-h_ind]
    features = feature_img.reshape(feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2])

    if train == True:
        ss_idx = subsample_idx(0, features.shape[0], num_examples)
        features = features[ss_idx]
    else:
        ss_idx = []

    h_features = harlick_features(np.pad(img_gray, h_ind, mode='constant'), h_neigh, ss_idx)
    features = np.hstack((features, h_features))

    if train == True:
        label = label[h_ind:-h_ind, h_ind:-h_ind]
        labels = label.reshape(label.shape[0]*label.shape[1], 1)
        labels = labels[ss_idx]
    elif trainval == True:
        #label = label[h_ind:-h_ind, h_ind:-h_ind]
        labels = label.reshape(label.shape[0]*label.shape[1], 1)
    else:
        labels = None

    # check how to combine features and labels?
    print(features.shape)
    print(labels.shape)
    return_dict_features[i] = features
    return_dict_labels[i] = labels

def create_features1(i, return_dict_features1, return_dict_labels, img, img_gray, img_lab, img_hsv, label, train=True, trainval=False):

    lbp_radius = 24 # local binary pattern neighbourhood
    h_neigh = 11 # haralick neighbourhood
    num_examples = 2000 # number of examples per image to use for training model

    lbp_points = lbp_radius*8
    h_ind = int((h_neigh - 1)/ 2)
    
    sobel_img_h = filters.sobel_h(img_gray)
    sobel_img_v = filters.sobel_v(img_gray)
    sobel_img = filters.sobel(img_gray)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
    lap_img = cv2.Laplacian(img, cv2.CV_64F)
    angle = np.zeros((img.shape[0], img.shape[1], 3))
    angle[grad_x!=0] = np.arctan(grad_y[grad_x!=0]/grad_x[grad_x!=0])
    angle[grad_x==0] = 90 

    feature_img = np.zeros((img.shape[0],img.shape[1], 23))
    feature_img[:,:,:3] = img
    feature_img[:,:,4:6] = img_hsv[:,:,1:3] # saturation and value channels from HSV color image.
    feature_img[:,:,6:9] = img_lab
    feature_img[:,:,9] = sobel_img_h # 56 did not have gradients
    feature_img[:,:,10] = sobel_img_v
    feature_img[:,:,11] = sobel_img
    feature_img[:,:,12] = gaussian_filter(img_gray, sigma=5)
    feature_img[:,:,13] = gaussian_filter(img_gray, sigma=3)
    feature_img[:,:,14:17] = lap_img
    feature_img[:,:,17:20] = np.sqrt(np.square(grad_x)+np.square(grad_y))
    feature_img[:,:,20:23] = angle

    img = None
    feature_img[:,:,3] = create_binary_pattern(img_gray, lbp_points, lbp_radius)
    if train==True: feature_img = feature_img[h_ind:-h_ind, h_ind:-h_ind]
    features = feature_img.reshape(feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2])

    if train == True:
        ss_idx = subsample_idx(0, features.shape[0], num_examples)
        features = features[ss_idx]
    else:
        ss_idx = []

    h_features = harlick_features(np.pad(img_gray, h_ind, mode='constant'), h_neigh, ss_idx)
    features = np.hstack((features, h_features))

    if train == True:
        label = label[h_ind:-h_ind, h_ind:-h_ind]
        labels = label.reshape(label.shape[0]*label.shape[1], 1)
        labels = labels[ss_idx]
    elif trainval == True:
        #label = label[h_ind:-h_ind, h_ind:-h_ind]
        labels = label.reshape(label.shape[0]*label.shape[1], 1)
    else:
        labels = None

    # check how to combine features and labels?
    print(features.shape)
    print(labels.shape)
    return_dict_features1[i] = features
    return_dict_labels[i] = labels

def create_training_dataset(image_list, label_list):

    print ('[INFO] Creating training dataset on %d image(s).' %len(image_list))

    X = []
    y = []
    
    #pool = mp.Pool(20)
    manager = mp.Manager()
    return_dict_features = manager.dict()
    return_dict_features1 = manager.dict()
    return_dict_labels = manager.dict()
    jobs = []

    for i, img in enumerate(image_list):
        print(label_list[i])
        # dividing jobs to save features into 2 dictionaries.
        if i<200: p=mp.Process(target=create_features, args=(i, return_dict_features, return_dict_labels, img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2LAB), cv2.cvtColor(img, cv2.COLOR_BGR2HSV), label_list[i]))
        if i>=200: p=mp.Process(target=create_features1, args=(i, return_dict_features1, return_dict_labels, img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2LAB), cv2.cvtColor(img, cv2.COLOR_BGR2HSV), label_list[i]))
        jobs.append(p)
        

    print(len(jobs))
    print(mp.cpu_count())

    for i in chunks(jobs, 50):
        for j in i:
            j.start()
        for j in i:
            j.join()

    X = np.vstack((np.array(list(return_dict_features.values())), np.array(list(return_dict_features1.values()))))
    y = np.array(list(return_dict_labels.values()))

    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    y = y.reshape(y.shape[0]*y.shape[1], y.shape[2]).ravel()

    print ('[INFO] Feature vector size:', X.shape)

    return X, y


def create_testing_dataset(image_list, label_list, model, output_dir):

    manager = mp.Manager()
    return_dict_features = manager.dict()
    return_dict_labels = manager.dict()
    jobs = []

    for i, img in enumerate(image_list):
        p=mp.Process(target=create_features, args=(i, return_dict_features, return_dict_labels, img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2LAB), cv2.cvtColor(img, cv2.COLOR_BGR2HSV), label_list[i], False, True))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    y = list(return_dict_labels.values())
    X = list(return_dict_features.values())

    iou = []
    for i in range(len(X)):
        import pdb; pdb.set_trace()
        try:
            class_iou, pred = test_model(np.squeeze(X[i], axis=0), np.squeeze(y[i], axis=0), model)
            pred = cv2.reshape(pred, (label_list[i].shape[1], label_list[i].shape[0]))
            cv2.imwrite(output_dir+'pred_'+str(i+1)+'.png', pred)
            cv2.imwrite(output_dir+'label_'+str(i+1)+'.png', label_list[i])
            cv2.imwrite(output_dir+'image_'+str(i+1)+'.jpg', image_list[i])
            iou.append(class_iou)

        except Exception as e:
            print(e)
    
    return iou


def train_model(X, y, classifier):

    if classifier == "SVM":
        from sklearn.svm import SVC
        print ('[INFO] Training Support Vector Machine model.')
        model = SVC()
        model.fit(X, y)
    elif classifier == "RF":
        from sklearn.ensemble import RandomForestClassifier
        print ('[INFO] Training Random Forest model.')
        rf_params = {
            'bootstrap': True, #notice the difference between estimators and fix that. use bootstrap False.
            'n_estimators': 20, 
            'random_state': 42, 
            'verbose': 2, 
            'n_jobs': 8, 
            #'max_features': 0.7,
            'max_depth': 15,  
            'min_samples_leaf': 12,
            'min_samples_split': 12, 
            'verbose': 0, 
            'warm_start': False,
            'min_weight_fraction_leaf': 0.0,
            'class_weight': 'balanced_subsample', #compute_class_weight("balanced", 4, y), 
            'criterion': 'gini'
        }

        model = RandomForestClassifier(**rf_params)
        model.fit(X, y)
    elif classifier == "GBC":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        model.fit(X, y)

    print ('[INFO] Model training complete.')
    print ('[INFO] Training Accuracy: %.2f' %model.score(X, y))
    return model


def test_model(X, y, model):

    pred = model.predict(X)
    precision = metrics.precision_score(y, pred, average='weighted', labels=np.unique(pred))
    recall = metrics.recall_score(y, pred, average='weighted', labels=np.unique(pred))
    f1 = metrics.f1_score(y, pred, average='weighted', labels=np.unique(pred))
    accuracy = metrics.accuracy_score(y, pred)

    print ('--------------------------------')
    print ('[RESULTS] Accuracy: %.2f' %accuracy)
    print ('[RESULTS] Precision: %.2f' %precision)
    print ('[RESULTS] Recall: %.2f' %recall)
    print ('[RESULTS] F1: %.2f' %f1)
    print ('--------------------------------')

    n_classes = 4
    class_iou = []
    for cclass in range(n_classes):
        groundtruth = y == cclass
        prediction = pred == cclass

        intersection = np.sum(groundtruth * prediction)
        union = np.sum(groundtruth + prediction)  

        iou = intersection/union.astype(np.float32) if union!=0 else 0  
        class_iou.append(iou) 

    print ('[RESULTS] Mean IoU: %.2f' %np.mean(class_iou))
    print ('[RESULTS] Mean IoU for each class: ')
    print ('Background class: ', class_iou[0])
    print ('Stem class: ', class_iou[1])
    print ('Callus class: ', class_iou[2])
    print ('Shoot class: ', class_iou[3])

    return class_iou, pred


def main(image_dir, label_dir, output_dir, classifier, test_resize, output_model_file):
    start = time.time()

    # Training dataset.
    print("[INFO] Prepare training dataset.")
    image_list, label_list = read_data(image_dir, label_dir)
    X_train, y_train = create_training_dataset(image_list, label_list)
    
    print("[INFO] Model training started.")
    model = train_model(X_train, y_train, classifier)
    print("[INFO] Model training complete.")
    
    # save the model.
    with open(output_model_file, "wb") as f:
        pkl.dump(model, f)
    print ('[INFO] Running inference for the training dataset.')
    test_model(X_train, y_train, model)
    
    if os.path.getsize(output_model_file) > 0:      
        with open(output_model_file, "rb") as f:
            unpickler = pkl.Unpickler(f)
            model = unpickler.load()
    print("model name: ", output_model_file)

    # Testing dataset.
    print ('\n--------------------------------')
    print("[INFO] Prepare testing dataset.")
    image_list, label_list = read_test_data(image_dir, label_dir, test_resize)
    iou = create_testing_dataset(image_list, label_list, model, output_dir)
  
    print ('[INFO] Running inference for the testing dataset.')
    iou = np.mean(iou, axis=0)
    print ('[RESULTS] Testing Mean IoU: %.2f' %np.mean(iou))
    print ('[RESULTS] Testing IoU for each class: ')
    print ('Background class: ', iou[0])
    print ('Stem class: ', iou[1])
    print ('Callus class: ', iou[2])
    print ('Shoot class: ', iou[3])

    print("\n[INFO] Time taken to complete the process. ", time.time()-start)


if __name__ == "__main__":
    args = parse_args()
    image_dir = args.image_dir
    label_dir = args.label_dir
    classifier = args.classifier
    output_dir = args.output_dir
    test_resize = args.test_resize
    output_model_file = args.output_model
    main(image_dir, label_dir, output_dir, classifier, test_resize, output_model_file)
