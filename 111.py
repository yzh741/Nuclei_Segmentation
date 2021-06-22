import utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import glob
from options import Options

# pred1 = np.array(Image.open('D:/multiorgan/1/test_same_segmentation/TCGA-21-5784-01Z-00-DX1_seg.tiff'))
# pred2 = np.array(Image.open('D:/multiorgan/1/watershed_process/TCGA-21-5784-01Z-00-DX1_seg.tiff'))
# gt = np.array(Image.open('D:/multiorgan/labels_instance/test_same/TCGA-21-5784-01Z-00-DX1.png'))
#
# result1 = utils.nuclei_accuracy_object_level(pred1, gt)
# aji_1 = result1[6]
# result2 = utils.nuclei_accuracy_object_level(pred2, gt)
# aji_2 = result2[6]
#
# print('aji_1:{}  aji_2:{}'.format(aji_1, aji_2))
# img = np.array(Image.open('D:/multiorgan/1/test_same_segmentation/TCGA-21-5784-01Z-00-DX1_seg.tiff'))
# h, w = img.shape
# binary = np.zeros((h, w))
# binary[img[:, :] > 0] = 255  # inside
# imageio.imsave('C:/Users/yzh/Desktop/8.png', binary)
# import openslide
#
# slide = openslide.OpenSlide('D:/eye_diseases/Data2020/data/4/svs/4_72.svs')
# print(slide.properties['aperio.MPP'])
# print(slide.level_dimensions)
# print(slide.level_downsamples)
# imgs = glob.glob('D:/eye_diseases/Data2020/data/4/svs/*.svs')
# count = 0
# for img in imgs:
#     slide = openslide.OpenSlide(img)
#     if slide.level_count < 6:
#         count = count + 1
#     print(slide.properties['aperio.MPP'])
#     #print('name:{},{}'.format(img, slide.level_count))
# print(count)

# img1 = np.array(slide.read_region((6041, 8068), 0, (1000, 1000)).convert('RGB'))
# img2 = np.array(slide.read_region((7041, 8068), 0, (1000, 1000)).convert('RGB'))
# img3 = np.array(slide.read_region((6041, 9068), 0, (1000, 1000)).convert('RGB'))
# img4 = np.array(slide.read_region((12915, 22403), 0, (1000, 1000)).convert('RGB'))
# imageio.imsave('D:/multiorgan/qulication_data/4_0/4_0_1.png', img1)
# imageio.imsave('D:/multiorgan/qulication_data/4_0/4_0_2.png', img2)
# imageio.imsave('D:/multiorgan/qulication_data/4_0/4_0_3.png', img3)
# imageio.imsave('D:/multiorgan/qulication_data/4_0/4_0_4.png', img4)
# plt.imshow(img1)
# plt.show()
# x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.uint8).transpose(
#     (1, 2, 0))
# print(x.shape)
# print(x[:, :, 0])
# imageio.imsave('C:/Users/yzh/Desktop/t.png', x)
# img = np.array(Image.open('C:/Users/yzh/Desktop/t.png'))
# print(img)
# print(img[:, :, 0])
import json
import pandas as pd
import seaborn as sns

# with open('C:/Users/yzh/Desktop/data.json') as f:
#     dicts = json.load(f)
#
# train_x = dicts['train_x']
# val_x = dicts['val_x']
# test_x = dicts['test_x']
# train_n = len(train_x)
# val_n = len(val_x)
# test_n = len(test_x)
#
# df = pd.DataFrame()
# df['data_split'] = ['train', 'valid', 'test']
# df['value'] = [train_n, val_n, test_n]
# print(df['value'])
# plt.ylim((0,700))
# p1 = sns.barplot(data=df, x='data_split', y='value')
#
# #plt.axis('off')
#
# for x, y in enumerate(df['value']):
#     plt.text(x-0.1, y+20, "%s" %y)
# plt.show()
# mix_img = imageio.imread('D:/multiorgan/mixed/TCGA-18-5592-01Z-00-DX1.png')
# he_img = imageio.imread('D:/multiorgan/H_stain/train/TCGA-18-5592-01Z-00-DX1.png')
# img = imageio.imread('D:/multiorgan/patches/images/clahe/588.png')
import cv2

from models.r2_unet import R2U_Net
import torch
import os
import openslide

# imgs = glob.glob('D:/multiorgan/qulication_data/best_8_59/normalize_segmentation/*.png')
# for path in imgs:
#     img = (np.array(Image.open(path)) * 255).astype(np.uint8)
#     name = os.path.basename(path).split('.')[0].split('_')
#     name = '_'.join(name[:3])
#     imageio.imsave(os.path.join('D:/multiorgan/qulication_data/best_8_59/colored', name+'.png'), img)
from my_transforms import get_transforms
from data_folder import DataFolder

# opt = Options(isTrain=True)
# opt.parse()
#
# data_transforms = {'train': get_transforms(opt.transform['train']),
#                    'val': get_transforms(opt.transform['val'])}
# dsets = {}
#
# img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], 'train')  # D:/multiorgan/images/train
# target_dir = '{:s}/{:s}'.format(opt.train['label_dir'], 'train')  # D:/multiorgan/labels/train
# weight_map_dir = '{:s}/{:s}'.format(opt.train['weight_map_dir'], 'train')  # D:/multiorgan/weight_maps/train
# dir_list = [img_dir, target_dir]
# if opt.dataset == 'multiorgan':
#     post_fix = ['label.png']
# else:
#     post_fix = ['anno_weight.png', 'anno.bmp']
# num_channels = [3, 3]  # 这个是干嘛的
# train_set = DataFolder(dir_list, post_fix, num_channels, data_transforms['train'])
#
# data = train_set[0]
# plt.subplot(121)
# plt.imshow(data[0])
# plt.subplot(122)
# plt.imshow(data[1])
# plt.show()
# img = Image.open('D:/multiorgan/H_stain/train/TCGA-18-5592-01Z-00-DX1.png')
# img1 = Image.open('D:/multiorgan/patches/data_expend2/multiorgan/images/train/0.png')
# labels = glob.glob('D:/multiorgan/patches/labels/train/*.png')
# for path in labels:
#     name = os.path.basename(path).split('.')[0]
#     os.rename(path, os.path.join('D:/multiorgan/patches/labels/train', name + '_label' + '.png'))
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import logging


def watershed_process(predict_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    masks = glob.glob(os.path.join(predict_dir, '*.tiff'))

    for i, path in enumerate(masks):
        name = os.path.basename(path).split('.')[0].split('_')
        name = '_'.join(name[:3])
        mask = np.array(Image.open(path))
        distance = ndi.distance_transform_edt(mask)
        local_maxi = peak_local_max(distance, labels=(mask > 0.6), min_distance=10, indices=False)
        markers = ndi.label(local_maxi)[0]
        image = watershed(-distance, markers, mask=mask)
        imageio.imsave(os.path.join(save_dir, name + '.tiff'), image)
        logging.info('name: {}, {}/{}'.format(name, i + 1, len(masks)))


root_path = 'D:/multiorgan/qulication_data/best_8/8-3'
dir_list = os.listdir(root_path)
# [best_2_3, 2,6]
logging.basicConfig(level=logging.INFO)
for dir in dir_list:
    slide_name = dir.split('_')
    slide_name = '_'.join(slide_name[1:])
    predict_dir = os.path.join(root_path, dir, slide_name + '_norm_segmentation')
    save_dir = os.path.join(root_path, dir, 'watershed')
    watershed_process(predict_dir, save_dir)

# img = Image.open('D:/oct/mask3/20200901104539_31000901_342_img2.png')
# img = np.array(img.resize((384, 384), Image.NEAREST))
# print(np.unique(img))

