import numpy as np
import queue
from matplotlib import pyplot as plt
from skimage import measure as smeasure
from skimage import io, transform, img_as_float
from scipy import ndimage
import cv2
import random

def unique_boxes(boxes, scale=1.0):
    """ return indices of unique boxes """
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep

def find_cls_box(sem_box):
    cls, cls_cnt = np.unique(sem_box, return_counts=True)

    if cls.shape[0] == 1:
        return cls[0]
    else:
        idx = np.argmax(cls_cnt[cls>0])
        return cls[cls>0][idx]

# 2019-06-16
def label2onehot(labelI, num_channel):
    ht, wd  = labelI.shape
    onehotM = np.zeros([ht, wd, num_channel])

    unq_ids = np.unique(labelI)
    for k in unq_ids:
        if k < num_channel:
            onehotM[labelI==k, k] = 1

    return onehotM

# 2019-05-31
def evaluate_instance_realV_mse(predM, gtM):
    '''
    evaluate the predict result by considering:
        the mse_intra_each_instance (to be -> 0) and
        inv_mse_inter_instances (mse to be >= 1).
    predM/gtM: in shape [bs, ht, wd, 1], [bs, ht, wd] or [ht, wd]
    '''
    # dealing with different input shape
    if len(predM.shape)== 2:
        predM = predM[np.newaxis, ...]
        gtM   = gtM[np.newaxis, ...]
    elif len(predM.shape) ==4:
        predM = predM[..., 0]
        gtM   = gtM[..., 0]
    elif len(predM.shape)<2 or len(predM.shape) > 4:
        print('input shape is invalid. support shape is: [bs,ht,wd,1], [bs,ht,wd], or [bs, wd]')
        return np.float('nan')
    predM_1d = np.reshape(predM, [predM.shape[0], -1])
    gtM_1d   = np.reshape(gtM, [gtM.shape[0], -1])

    # compute mse
    bs, length = predM_1d.shape
    pair_count = length*2 - 3
    inter_mse, intra_mse = 0, 0
    idx = np.arange(length)
    for bk in range(bs):
        np.random.shuffle(idx)

        gt_label  = (gtM_1d[bk, idx[:-1]] == gtM_1d[bk, idx[1:]]).astype(np.float)
        pred_diff = np.abs(predM_1d[bk, idx[:-1]] - predM_1d[bk, idx[1:]])

        gt_label_2  = (gtM_1d[bk, idx[:-2]] == gtM_1d[bk, idx[2:]]).astype(np.float)
        pred_diff_2 = np.abs(predM_1d[bk, idx[:-2]] == predM_1d[bk, idx[2:]])

        intra_mse += (gt_label*np.power(np.minimum(pred_diff, 1.0), 2) + \
                        gt_label_2*np.power(np.minimum(pred_diff_2, 1.0), 2))/(pair_count)
        inter_mse += ((1-gt_label)*np.power(np.minimum(pred_diff, 1.0), 2) + \
                        (1-gt_label_2)*np.power(np.minimum(pred_diff_2, 1.0), 2))/(pair_count)

    return [intra_mse, inter_mse]


# 2019-05-21
def map_instanceID_semID(instM, semM, return_inst_info=False, ignore_label=255):
    '''
    Find corresponding semantic ID for each instance in instM.

    instM / semM: in shape [bs, ht, wd, 1], [bs, ht, wd] or [ht, wd]
    '''
    # dealing with different input shape
    if  isinstance(instM, (list,)):
        bs = len(instM)
        if len(instM[0].shape)==3:
            for k in range(bs):
                instM[k] = instM[k][...,0]
                semM[k]  = semM[k][...,0]
        if len(instM[0].shape)>3 or len(instM[0].shape)<2:
            print('input shape is invalid. support shape is: [bs,ht,wd,1], [bs,ht,wd], or [bs, wd]')
            return [{0:0}]
    else:
        bs = instM.shape[0]
        if len(instM.shape)== 2:
            instM = instM[np.newaxis, ...]
            semM  = semM[np.newaxis, ...]
        elif len(instM.shape) ==4:
            instM = instM[..., 0]
            semM  = semM[..., 0]
        elif len(instM.shape)<2 or len(instM.shape) > 4:
            print('input shape is invalid. support shape is: [bs,ht,wd,1], [bs,ht,wd], or [bs, wd]')
            return [{0:0}]

    # after preprocess, instM/semM in shape [bs, ht, wd]
    mapping_lut = [None]*bs
    for bk in range(bs):
        mapping_lut[bk] = dict()
        inst_props = smeasure.regionprops(instM[bk])
        for prop in inst_props:
            if prop.label == ignore_label:
                continue
            coord = prop.coords
            cov_sem_ids, cov_sem_cnts = np.unique(semM[bk][coord[:,0], coord[:,1]], return_counts=True)
            sem_id, sem_cnt = 0, 0
            for si, sc in zip(cov_sem_ids, cov_sem_cnts):
                if si ==0 or si == ignore_label:
                    continue
                elif sem_cnt < sc:
                    sem_id, sem_cnt = si, sc

            if return_inst_info is False:
                mapping_lut[bk][prop.label] = sem_id
            else:
                mapping_lut[bk][prop.label] = {'sem': sem_id, 'box': prop.bbox}

    return mapping_lut


# 2019-04-15
def resize_labelI(inLabel, newShape):
    return cv2.resize(inLabel, (newShape[1], newShape[0]), interpolation=cv2.INTER_NEAREST)

def resize_map(inArr, newShape, method='transfer'):
    '''
    inArr: support multiple channels.
           doesn't support scale of label image
    newShape: (ht, wd)
    '''
    if method == 'transform':
        resized_normalized = transform.resize((inArr - inArr.min()) / (inArr.max() - inArr.min()),
                                            (newShape[0], newShape[1]), order=3, mode='reflect')
        resized_denormalized = (resized_normalized * (inArr.max() - inArr.min())) + inArr.min()
        return resized_denormalized
    elif method == 'ndimage':
        if len(inArr.shape) == 2:
            resized_map = ndimage.zoom(inArr, (float(newShape[0])/inArr.shape[0], float(newShape[1])/inArr.shape[1]), order=3, prefilter=False)
        else:
            resized_map = ndimage.zoom(inArr, (float(newShape[0])/inArr.shape[0], float(newShape[1])/inArr.shape[1], 1), order=3, prefilter=False)
        return resized_map
    else: # cv2.
        # support inArr with channel =3 or 1.
        resized_map = cv2.resize(inArr, (newShape[1], newShape[0]))
        return resized_map



# 2018-11-13
# add for data augmentation.
def preprocess_data_augmentation(rgbI, semI, instI, aug_options,
                                  min_cropsize=[32,32], ignore_label=255):
    '''
    rgbI: [ht, wd, 3]
    semI/instI: [ht, wd]
    aug_options: might include ['hflip', 'vflip', 'random crop', 'jitter crop', 'rotation']
                 *** 'random crop' and 'jitter crop' can only have one.
    '''

    def _random_crop_box(img_ht, img_wd, min_cropsize):
        crop_ratio   = np.random.uniform(0.5, 1.0)

        crop_vsize   = max(min_cropsize[0], int(crop_ratio*img_ht))
        crop_hsize   = max(min_cropsize[1], int(crop_ratio*img_wd))
        max_oft_ht   = img_ht - crop_vsize
        max_oft_wd   = img_wd - crop_hsize
        oft_ht       = 0 if max_oft_ht==0 else np.random.randint(0, max_oft_ht)
        oft_wd       = 0 if max_oft_wd==0 else np.random.randint(0, max_oft_wd)

        return [oft_ht, oft_wd, oft_ht+crop_vsize, oft_wd+crop_hsize]

    def _jitter_crop_box(instI, min_cropsize, ignore_label):
        props = smeasure.regionprops(instI*(instI!=ignore_label))
        if len(props) == 0:
            return  _random_crop_box(instI.shape[0], instI.shape[1], min_cropsize)
        else:
            prop = random.choice(props)
            y0, x0, y1, x1 = prop.bbox
            b_ht, b_wd     = y1-y0, x1-x0
            y0             = 0 if y0==0 else np.random.randint(max(0, y0-0.3*b_ht), y0)
            x0             = 0 if x0==0 else np.random.randint(max(0, x0-0.3*b_wd), x0)

            sc_ratio       = np.random.uniform(1.3, 1.8)
            b_ht           = max(int(sc_ratio*b_ht), min_cropsize[0])
            b_wd           = max(int(sc_ratio*b_wd), min_cropsize[1])
            # keep ht/wd ratio
            b_ht = max(b_ht, int(instI.shape[0]*b_wd/instI.shape[1])+1)
            b_wd = max(b_wd, int(instI.shape[1]*b_ht/instI.shape[0])+1)

            if(y0+b_ht>=instI.shape[0]):
                y0 = max(0, instI.shape[0]-b_ht)
                y1 = instI.shape[0]
            else:
                y1 = y0+b_ht

            if(x0+b_wd>=instI.shape[1]):
                x0 = max(0, instI.shape[1]-b_wd)
                x1 = instI.shape[1]
            else:
                x1 = x0+b_wd

            return [y0, x0, y1, x1]

    # main process
    # flip
    if 'hflip' in aug_options:
        rgbI  = np.fliplr(rgbI)
        semI  = np.fliplr(semI)
        instI = np.fliplr(instI)
    if 'vflip' in aug_options:
        rgbI  = np.flipud(rgbI)
        semI  = np.flipud(semI)
        instI = np.flipud(instI)
    # crop
    if 'random crop' in aug_options or 'jitter crop' in aug_options:
        all_inst_ids, all_inst_cnts = np.unique(instI, return_counts=True)
        all_inst_lut = dict()
        for uid, cnt in zip(all_inst_ids, all_inst_cnts):
            all_inst_lut[uid] = cnt

        k = 0
        while(k<5):
            # ramdom crop
            if 'random crop' in aug_options:
                y0, x0, y1, x1 = _random_crop_box(rgbI.shape[0], rgbI.shape[1], min_cropsize)
            else: # 'jitter crop' in aug_options:
                y0, x0, y1, x1 = _jitter_crop_box(instI, min_cropsize, ignore_label)
            crop_instI  = instI[y0:y1, x0:x1]
            crop_semI   = semI[y0:y1, x0:x1]

            # check crop quality
            crop_inst_ids, crop_inst_cnts = np.unique(crop_instI, return_counts=True)
            for uid, cnt in zip(crop_inst_ids, crop_inst_cnts):
                if uid!=0 and cnt < all_inst_lut[uid]*0.7:
                    crop_instI[crop_instI==uid] = 0
                    crop_semI[crop_instI==uid] = 0
            if np.max(crop_instI)>0:
                break
            else:
                k += 1
                if k == 4:
                    aug_options[-1] = 'jitter crop'


        rgbI   = rgbI[y0:y1, x0:x1,:]
        semI   = crop_semI
        instI  = crop_instI

    # rotation
    if 'rotation' in aug_options and np.random.binomial(1, 0.5)==1:
        angle = np.random.uniform(-10, 10)
        print('Rotation is not implemented. Please do not select it')

    return rgbI, semI, instI


# 2019-03-12
# add for instance segmentation checking of boundary precision
def prediction_boundary_check(in_predI, in_gtI):
    predI = in_predI.astype(np.int)
    gtI   = in_gtI.astype(np.int)
    predI[gtI==255] = 0
    gtI[gtI==255] = 0


    edge_gtI = cv2.Canny(gtI.astype(np.uint8), 0.1, 1)

    false_colorI = np.zeros([predI.shape[0], predI.shape[1], 3])
    labelI = np.ones([predI.shape[0], predI.shape[1]])*3

    pred_props = smeasure.regionprops(predI)
    scale_red = 255./(1+len(pred_props))

    gt_props = smeasure.regionprops(gtI)
    scale_blue = 255./(1+len(gt_props))

    scale_green = 255./(1+(len(pred_props)+1)*len(gt_props))

    for k in range(len(pred_props)):
        coord = pred_props[k].coords
        false_colorI[coord[:,0], coord[:,1], 0] = (k+1) * scale_red
        false_colorI[coord[:,0], coord[:,1], 0] = (k+1) * scale_red
        labelI[coord[:,0], coord[:,1]] = 1

    for k in range(len(gt_props)):
        coord = gt_props[k].coords
        for py, px in zip(coord[:,0], coord[:,1]):
            false_colorI[py,px, 1] = 0  if false_colorI[py,px,0]==0 else false_colorI[py, px, 0]
            false_colorI[py,px, 2] = (k+1)*scale_blue if false_colorI[py,px,0]==0 else false_colorI[py, px, 0]
            labelI[py,px] = 2 if labelI[py, px]==0 else 3

    labelI[edge_gtI>0] = 0
    false_colorI[edge_gtI>0, 1] = 255

    return false_colorI.astype(np.uint8), labelI.astype(np.uint8)


# 2018-10-17
# add for connectivity-label union box detection.
def _check_box_overlap(box1, box2):
    y0 = max(box1[0], box2[0])
    x0 = max(box1[1], box2[1])
    y1 = min(box1[2], box2[2])
    x1 = min(box1[3], box2[3])

    return (y1>=y0 and x1>=x0)

def _find_overlap_boxes(idx, overlap_mat, seeds, mode='ver'):
    if mode is 'ver':
        for k in range(overlap_mat.shape[0]):
            if overlap_mat[k, idx]==1:
                seeds.put([k, 'hor'])
        overlap_mat[:,idx] = 0
    else:
        for k in range(overlap_mat.shape[1]):
            if overlap_mat[idx, k]==1:
                seeds.put([k, 'ver'])
        overlap_mat[idx, :] = 0


def group_union_boxes(boxes_1, boxes_2):
    num_1, num_2 = len(boxes_1), len(boxes_2)
    # compute overlap
    ov_mat = np.zeros((num_1, num_2))
    for j in range(num_1):
        for i in range(num_2):
            ov_mat[j, i] = _check_box_overlap(boxes_1[j], boxes_2[i])

    # group union boxes
    union_boxes = {}
    cnt = 1
    processed = [False]*num_1
    for j in range(num_1):
        if processed[j]:
            continue

        y0,x0,y1,x1 = boxes_1[j]
        seeds       = queue.Queue()
        seeds.put([j, 'hor'])
        while seeds.qsize()>0:
            idx, mode       = seeds.get()
            _find_overlap_boxes(idx, ov_mat, seeds, mode)
            ry0,rx0,ry1,rx1 = boxes_1[idx] if mode is 'hor' else boxes_2[idx]
            y0, x0          = min(y0, ry0), min(x0, rx0)
            y1, x1          = max(y1, ry1), max(x1, rx1)
            if mode is 'hor':
                processed[idx] = True

        union_boxes[cnt] = [y0,x0,y1,x1]
        cnt += 1

    return union_boxes

def test_group_union_boxes():
    # test group_union_boxes.
    img  = np.zeros([512, 512, 3], dtype=np.uint8)
    boxes_1 = [[30,  5,  60, 39],
               [70, 40, 100, 75],
               [8,  70, 30,  95],
               [50, 90, 70,  100],
               [200, 200, 300, 300],
               [350, 200, 450, 450] ]
    boxes_2 = [[10,  30,  50, 75],
               [55,  30,  95,  60],
               [ 5,  70,  93,  100],
               [200, 200, 300, 300],
               [350, 200, 450, 450] ]

    for k in range(len(boxes_1)):
        y0,x0,y1,x1 = boxes_1[k]
        img[y0:y1+1, x0:x1+1, 0] = 255
    for k in range(len(boxes_2)):
        y0,x0,y1,x1 = boxes_2[k]
        img[y0:y1+1, x0:x1+1, 1] = 255

    union_boxes = group_union_boxes(boxes_1, boxes_2)
    img2 = np.zeros([512, 512], dtype = np.uint8)
    for k in union_boxes:
        y0,x0,y1,x1 = union_boxes[k]
        img2[y0:y1+1, x0:x1+1] = 255

    fig, ax = plt.subplots(2)
    ax[0].imshow(img)
    ax[1].imshow(img2)
    plt.show()


if __name__ == '__main__':
    # test_group_union_boxes()

    fpath = '../../Cityscapes/Code/gtFine/val/frankfurt/'
    fname = 'frankfurt_000001_050149_gtFine_instanceTrainIds.png'

    import scipy.misc as smisc
    I = smisc.imread(fpath+fname, mode='P')

    if False:
        fc_colorI, labelI = prediction_boundary_check(I[100:, 200:], I[:-100, :-200])
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(fc_colorI)
        ax[1].imshow(labelI)
        plt.show()
    else:
        gtM        = I
        sigma_list = [0.01, 0.1, 0.3, 1.0]
        for sigma in sigma_list:
            predM = np.random.normal(0, sigma, gtM.shape)
            inter_mse, intra_mse = evaluate_instance_realV_mse(predM, gtM)

            print("sigma = {:.3f}, inter_mse={:.5f}, intra_mse={:.5f}".format(inter_mse, intra_mse))
