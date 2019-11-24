from collections import defaultdict
import numpy as np
import scipy.misc as smisc

from matplotlib import pyplot as plt

class ColorMap:
    def __init__(self, label_num=21):
        self.inv_cmap = defaultdict()
        self.cmap     = np.zeros([label_num, 3], dtype=np.uint8)
        for i in range(label_num):
            k, r,g,b = i, 0,0,0
            for j in range(8):
                r |= (k&1)      << (7-j)
                g |= ((k&2)>>1) << (7-j)
                b |= ((k&4)>>2) << (7-j)
                k = k>>3

            self.cmap[i,:] = [r,g,b]
            inv_key = (((r<<8) + g)<<8) + b
            self.inv_cmap[inv_key] = i


    def convert_label2rgb(self, inI):
        ht, wd = inI.shape
        rgbI = np.zeros([ht, wd, 3], dtype = np.uint8)
        rgbI[:,:,0] = self.cmap[inI, 0]
        rgbI[:,:,1] = self.cmap[inI, 1]
        rgbI[:,:,2] = self.cmap[inI, 2]

        return rgbI

    def get_colormap(self):
        return self.cmap/255.0


    def convert_rgb2label(self, inI):
        ht, wd, _ = inI.shape
        grayI = np.zeros([ht, wd], dtype=np.uint8)

        inI   = inI.astype(np.int)
        keyI  = (((inI[:,:,2]<<8) + inI[:,:,1])<<8) + inI[:,:,0]
        keyList = np.unique(keyI)
        for ele in keyList:
            grayI[keyI==ele] = self.inv_cmap[ele]

        return grayI


if __name__ == '__main__':

    import pdb
    pdb.set_trace()

    I = smisc.imread('/home/yuanjial/Documents/DataSet/PASCAL/sourceData/SegmentationObjectFilledDenseCRF/2007_000323.png', mode='P')

    cm = ColorMap(256)
    rgbI = cm.convert_label2rgb(I)
    invI = cm.convert_rgb2label(rgbI)

    plt.imshow(rgbI)
    plt.show()






