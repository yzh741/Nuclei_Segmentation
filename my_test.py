import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
import imageio
import os
import numpy as np
import glob
from PIL import Image
from data_folder import get_imgs_list
import matplotlib.pyplot as plt
import logging


# 线下增强
def main(img_dir, weight_dir, label_dir, pos_fix):
    img_list = get_imgs_list([img_dir, weight_dir, label_dir], pos_fix)
    print(img_list)
    rotate90 = iaa.Rot90(1)
    rotate180 = iaa.Rot90(2)
    rotate270 = iaa.Rot90(3)
    horizontal = iaa.Fliplr(1.0)
    vertical = iaa.Flipud(1.0)
    for i, image in enumerate(img_list):
        img_path, weight_path, label_path = image
        name = os.path.basename(img_path).split('.')[0]
        img = np.array(Image.open(img_path))
        weight_map = np.array(Image.open(weight_path))
        label = np.array(Image.open(label_path))
        img = [img, weight_map, label]
        # segmaps = SegmentationMapsOnImage(np.array(Image.open(label_path)), shape=img.shape)
        img_90 = rotate90(images=img)  # 通过控制parameter image/images 来控制单图片还是多图片
        img_180 = rotate180(images=img)
        img_270 = rotate270(images=img)
        img_h = horizontal(images=img)
        img_v = vertical(images=img)

        imageio.imwrite(os.path.join(img_dir, name + '_rot90' + '.png'), img_90[0])
        imageio.imwrite(os.path.join(img_dir, name + '_rot180' + '.png'), img_180[0])
        imageio.imwrite(os.path.join(img_dir, name + '_rot270' + '.png'), img_270[0])
        imageio.imwrite(os.path.join(img_dir, name + '_horizontal' + '.png'), img_h[0])
        imageio.imwrite(os.path.join(img_dir, name + '_vertical' + '.png'), img_v[0])

        imageio.imwrite(os.path.join(weight_dir, name + '_rot90' + '_weight' + '.png'), img_90[1])
        imageio.imwrite(os.path.join(weight_dir, name + '_rot180' + '_weight' + '.png'), img_180[1])
        imageio.imwrite(os.path.join(weight_dir, name + '_rot270' + '_weight' + '.png'), img_270[1])
        imageio.imwrite(os.path.join(weight_dir, name + '_horizontal' + '_weight' + '.png'), img_h[1])
        imageio.imwrite(os.path.join(weight_dir, name + '_vertical' + '_weight' + '.png'), img_v[1])

        imageio.imwrite(os.path.join(label_dir, name + '_rot90' + '_label' + '.png'), img_90[2])
        imageio.imwrite(os.path.join(label_dir, name + '_rot180' + '_label' + '.png'), img_180[2])
        imageio.imwrite(os.path.join(label_dir, name + '_rot270' + '_label' + '.png'), img_270[2])
        imageio.imwrite(os.path.join(label_dir, name + '_horizontal' + '_label' + '.png'), img_h[2])
        imageio.imwrite(os.path.join(label_dir, name + '_vertical' + '_label' + '.png'), img_v[2])
        logging.info('name:{},  {}/{}'.format(name, i + 1, len(img_list)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    img_dir = 'D:/multiorgan/patches/images/train'
    label_dir = 'D:/multiorgan/patches/labels/train'
    weight_dir = 'D:/multiorgan/patches/weight_maps/train'
    pos_fix = ['weight.png', 'label.png']
    main(img_dir, weight_dir, label_dir, pos_fix)

