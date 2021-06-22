from utils import color
import glob
import logging
import os
import torch
import numpy as np
from PIL import Image
import PIL
import utils
import matplotlib.pyplot as plt
from prepare_data import create_folder
from my_transforms import get_transforms
from data_folder import DataFolder
from options import Options


def main(img_list, save_dir):
    all_results = {}
    for i, img_path in enumerate(img_list):
        name = os.path.basename(img_path).split('.')[0]
        # mode = name.split('_')[2]
        img = np.array(Image.open(img_path))
        result = calculation(img)
        identity = int(name.split('_')[0])
        all_results[name] = tuple([*result, identity])
    header = ['count', 'cell_area', 'density', 'identity']
    save_results(header, all_results, os.path.join(save_dir, 'result.txt'))


def calculation(img, mpp=0.241876):
    # 核的个数，面积就是非0区域，密度 0.241876um/pixel
    per_pixel_area = mpp * mpp  # 单个像素的面积 um^2
    h, w = img.shape[0], img.shape[1]
    count = np.max(img).astype(float)  # number of nuclei
    non_zero_count = h * w - (img == 0).sum()
    cell_area = non_zero_count * per_pixel_area
    total_area = h * w * per_pixel_area
    density = count / total_area  # per  / um^2
    return [count, cell_area, density]


def save_results(header, all_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    # assert N == len(avg_results)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # all results
        for key, values in sorted(all_results.items()):
            file.write('{:s}:'.format(key))
            for value in values:
                file.write('\t{:.8f}'.format(value))
            file.write('\n')


if __name__ == '__main__':
    root_path = 'D:/multiorgan/qulication_data/best_8/8-3'
    dir_list = os.listdir(root_path)
    for dir in dir_list:
        img_list = glob.glob(os.path.join(root_path, dir, 'watershed', '*.tiff'))
        save_dir = os.path.join(root_path, dir)
        main(img_list, save_dir)
    # img_list = glob.glob('D:/multiorgan/MoNuSegTestData/labels_instance/*.png')
    # save_dir = 'D:/multiorgan/MoNuSegTestData/colored'
    #pred_labeled = np.array(Image.open('D:/multiorgan/MoNuSegTestData/labels_instance'))
    # create_folder(save_dir)
    # for img_path in img_list:
    #     pred_colored = np.zeros((1000, 1000, 3))
    #     name = os.path.basename(img_path).split('.')[0]
    #     pred_labeled = np.array(Image.open(img_path))
    #     for k in range(1, pred_labeled.max() + 1):
    #         pred_colored[pred_labeled == k, :] = np.array(utils.get_random_color())
    #     pred_colored_pil = Image.fromarray((pred_colored*255).astype(np.uint8))
    #     pred_colored_pil.save(os.path.join(save_dir, name + '.png'))

