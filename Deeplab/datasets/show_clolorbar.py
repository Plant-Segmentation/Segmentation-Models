import sys
sys.path.append('../')

import numpy as np
from PIL import Image
from PIL import ImageDraw
from plant import palette
from config_plant import config as cfg


def show_color_bar():
    # define size
    blk_ht, blk_wd = 20, 30
    wh_bar_margin  = 3
    text_wd        = 80
    ht  = blk_ht*cfg.num_classes+wh_bar_margin*(cfg.num_classes+1)
    wd  = 2*wh_bar_margin + blk_wd + text_wd

    # save
    save_img = Image.new('RGB', (wd, ht))
    draw_img = ImageDraw.Draw(save_img)
    save_img.paste(Image.fromarray(np.ones([ht, wd, 3], np.uint8)*255))
    for k in range(cfg.num_classes):
        color_blk        = np.ones([blk_ht, blk_wd, 3], np.uint8)
        color_blk[...,:] = cfg.save_color[k]

        st_ht = wh_bar_margin + (wh_bar_margin + blk_ht) * k
        save_img.paste(Image.fromarray(color_blk), (wh_bar_margin, st_ht))
        draw_img.text((2*wh_bar_margin+blk_wd, st_ht+(blk_ht/3.0)), cfg.class_name[k], fill=(0,0,0))

    save_img.save(cfg.label_path+'.png')


if __name__ == '__main__':
    show_color_bar()


