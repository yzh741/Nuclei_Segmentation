from prepare_data import create_folder
import math
import os
import argparse
import logging
import numpy as np
from PIL import Image

parse = argparse.ArgumentParser(description='split image in to patch')
parse.add_argument('img_dir', default=None, metavar='ANCHOR_IMG', type=str, help='Path to the color_norm dir')
parse.add_argument('label_dir', default=None, metavar='IMAGE_DIR', type=str, help='Path to the label dir')
parse.add_argument('weightmap_dir', default=None, metavar='WEIGHT_MAP', type=str,
                   help='Path to the weightmap dir')


def split_patches(data_dir, save_dir, postfix=None):
    """ split large image into small patches """
    # 根据data_dir地址确定save_dir
    create_folder(save_dir)
    mode = os.path.basename(data_dir)  # train/val

    save_dir = os.path.join(save_dir, mode)
    create_folder(save_dir)

    image_list = os.listdir(data_dir)
    for i_, image_name in enumerate(image_list):
        name = image_name.split('.')[0]
        if postfix and name[-len(postfix):] != postfix:  # label
            continue
        image_path = os.path.join(data_dir, image_name)
        image = np.array(Image.open(image_path))
        # image = io.imread(image_path)
        seg_imgs = []

        # split into 16 patches of size 250x250
        h, w = image.shape[0], image.shape[1]
        patch_size = 250
        h_overlap = math.ceil((4 * patch_size - h) / 3)
        w_overlap = math.ceil((4 * patch_size - w) / 3)
        for i in range(0, h - patch_size + 1, patch_size - h_overlap):
            for j in range(0, w - patch_size + 1, patch_size - w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i + patch_size, j:j + patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)


        for k in range(len(seg_imgs)):

            if postfix:
                seg_imgs_pil = Image.fromarray(seg_imgs[k])
                seg_imgs_pil.save('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(postfix) - 1], k, postfix))
                # io.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(postfix) - 1], k, postfix), seg_imgs[k])
            else:
                seg_imgs_pil = Image.fromarray(seg_imgs[k])
                seg_imgs_pil.save('{:s}/{:s}_{:d}.png'.format(save_dir, name, k))
                # io.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])
            logging.info('{},{}/{}'.format(name, i_+1, k+1))


def main():
    args = parse.parse_args()
    logging.basicConfig(level=logging.INFO)
    img_dir = args.img_dir
    label_dir = args.label_dir
    weightmap_dir = args.weightmap_dir
    patch_folder = '{:s}/patches'.format('D:/multiorgan')  # 存放patch
    create_folder(patch_folder)
    split_patches(img_dir, '{:s}/images'.format(patch_folder))  # 切染色均一化图片
    split_patches(label_dir, '{:s}/labels'.format(patch_folder), 'label')  # 切三元mask
    split_patches(weightmap_dir, '{:s}/weight_maps'.format(patch_folder), 'weight')  # 切权重mask


if __name__ == '__main__':
    main()
