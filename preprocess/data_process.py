#
# Importing all the necessary libraries
#

import os
import numpy as np
import skimage
from skimage.exposure import equalize_adapthist
from skimage import img_as_float
import imageio
from skimage.transform import resize as skimageresize
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import cv2
from imgaug import augmenters as iaa

#
# Setting all the global parameters
#

im_height = 1000
im_width = 1000
channels = 4
patch_size_image = 250
patch_size_model = 208
height = patch_size_image
width = patch_size_image
stride = 125  # 50 overlap

#
# Defining paths to different image folders
#

# Base path:
base_path = 'D:/multiorgan/'

# Pre-processed Images directory path:
im_path = base_path + 'Tissue_images/train/'
# Pre-processed Target directory path:
# target_path = base_path + 'train/'
# Pre-processed Target where 50% border is removed, directory path:
# target_border_rem_50_path = base_path + 'target_border_remove_50/'
# Pre-processed Target have tenary class
target_tenary_path = base_path + 'labels/train/'
# Processed original size images path:
im_processed_path = base_path + 'images/train/'
# H Stain images path (sklearn rgb2hed)
h_stain_path = base_path + 'H_stain/train/'

# save_dir = 'D:/multiorgan/patches/val'

#
# Reading the normalized tissue images stored on disk and storing them as array
#

all_files_temp = os.listdir(im_path)
all_files = []
for file in tqdm(all_files_temp):
    name = os.path.splitext(file)[0]
    all_files.append(name)

print(all_files)  # name
original_images = np.zeros((len(all_files), im_height, im_width), dtype=np.uint8)
print(original_images.shape)

# 3通道染色归一化图叠加，H&E染色的H通道

# for im_i in tqdm(range(len(all_files))):
#     image_file = im_processed_path + all_files[im_i] + '.png'
#     original_images[im_i, :, :, :3] = imageio.imread(image_file)  # load normalize image

for im_i in tqdm(range(len(all_files))):
    image_file = h_stain_path + all_files[im_i] + '.png'
    original_images[im_i, :, :] = imageio.imread(image_file)

# for i, name in enumerate(all_files):
#     imageio.imsave(os.path.join('D:/multiorgan/mixed/test_diff', name + '.png'), original_images[i, :, :, :])


# Reading the original masks stored on disk and storing them as array

original_masks = np.zeros((len(all_files), im_height, im_width, 3))
for im_i in tqdm(range(len(all_files))):
    mask_file = target_tenary_path + all_files[im_i] + '_label.png'
    original_masks[im_i] = imageio.imread(mask_file)

print(original_masks.shape)

# Creating patches of defined size from the full size images/masks and storing them as array
# (with fixed strides and no augmentation)

num_files = len(all_files)
num_images = int(num_files * (np.floor((im_height - height) / stride) + 1) ** 2)

print('Number of files: %d' % num_files)
print('Number of images total: %d' % num_images)

# Creating patches of defined size from the full size images/masks and storing them as array (by randomly picking a patch from the full image and with augmentation)


num_files = len(all_files)
# original_per_image = 500
# augmented_1_per_image = 300
# augmented_2_per_image = 50  # CLAHE
# augmented_3_per_image = 300
augmented_4_per_image = 50
augmented_5_per_image = 50
augmented_6_per_image = 50
# augmented_7_per_image = 150
num_images_original = int(num_files * (np.floor((im_height - height) / stride) + 1) ** 2)
num_images_augmented = (augmented_6_per_image +
                        augmented_4_per_image +
                        augmented_5_per_image) * num_files
num_images = num_images_original + num_images_augmented

print('Number of files: %d' % num_files)
print('Number of images original: %d' % num_images_original)
print('Number of images augmented: %d' % num_images_augmented)
print('Number of images total: %d' % (num_images_original + num_images_augmented))

images = np.zeros((num_images, patch_size_image, patch_size_image), dtype=np.uint8)
masks = np.zeros((num_images, patch_size_image, patch_size_image, 3), dtype=np.uint8)

print(images.shape)
print(masks.shape)
counter = 0

for im_i in tqdm(range(len(all_files))):
    image = original_images[im_i]
    mask = original_masks[im_i]
    for i in list(range(0, im_height - height + 1, stride)):
        for j in list(range(0, im_width - height + 1, stride)):
            images[counter, :, :] = image[i:i + height, j:j + width]
            masks[counter, :, :, :] = mask[i:i + height, j:j + width]
            counter += 1

seq = iaa.Sequential([iaa.Fliplr(0.15), iaa.Flipud(0.15)])
seq_det = seq.to_deterministic()
images = seq_det.augment_images(images)
masks = seq_det.augment_images(masks)

for i in range(images.shape[0]):
    imageio.imsave(os.path.join('D:/multiorgan/patches/images/train2', str(i) + '.png'), images[i, :, :])
    imageio.imsave(os.path.join('D:/multiorgan/patches/labels/train2', str(i) + '.png'), masks[i, :, :, :])

# print('Augmentation 2 : CLAHE')
#
#
# def augment_clahe(img):
#     img_adapteq = equalize_adapthist(img, clip_limit=0.03)
#     return img_adapteq
#
#
# for im_i in tqdm(range(len(all_files))):
#     image = original_images[im_i]
#     mask = original_masks[im_i]
#     y = random.sample(range(0, im_height - patch_size_image + 1), augmented_2_per_image)
#     x = random.sample(range(0, im_width - patch_size_image + 1), augmented_2_per_image)
#     for i in range(augmented_2_per_image):
#         image_patch = image[y[i]:y[i] + patch_size_image, x[i]:x[i] + patch_size_image, :]
#         mask_patch = mask[y[i]:y[i] + patch_size_image, x[i]:x[i] + patch_size_image, :]
#         image_patch2 = augment_clahe(image_patch[:, :, :3])  # 只对normalize做增强，另一个channel不管
#         images[counter, :, :, :3] = np.round(image_patch2 * 255)
#         images[counter, :, :, 3] = image_patch[:, :, 3]
#         masks[counter, :, :, :] = mask_patch
#         counter += 1

# print(images[588, :, :, 0].dtype)
# print(images[588, :, :, 0])
# for i in range(num_images_original, counter):
#     imageio.imsave(os.path.join('D:/multiorgan/patches/images/clahe', str(i) + '.png'), images[i, :, :, :])
#     imageio.imsave(os.path.join('D:/multiorgan/patches/labels/clahe', str(i) + '.png'), masks[i, :, :, :])


print('Augmentation 4 : GaussianBlur')
augmenter4 = iaa.GaussianBlur(1.0)
start = counter
for im_i in tqdm(range(len(all_files))):
    image = original_images[im_i]
    mask = original_masks[im_i]
    y = random.sample(range(0, im_height - patch_size_image + 1), augmented_4_per_image)
    x = random.sample(range(0, im_width - patch_size_image + 1), augmented_4_per_image)
    for i in range(augmented_4_per_image):
        image_patch = image[y[i]:y[i] + patch_size_image, x[i]:x[i] + patch_size_image]
        mask_patch = mask[y[i]:y[i] + patch_size_image, x[i]:x[i] + patch_size_image, :]
        images[counter, :, :] = augmenter4.augment_image(image_patch)
        masks[counter, :, :, :] = mask_patch
        counter += 1

for i in range(start, counter):
    imageio.imsave(os.path.join('D:/multiorgan/patches/images/train2', str(i) + '.png'), images[i, :, :])
    imageio.imsave(os.path.join('D:/multiorgan/patches/labels/train2', str(i) + '.png'), masks[i, :, :, :])

start = counter
print('Augmentation 5 : Affine Rotate')
for im_i in tqdm(range(len(all_files))):
    image = original_images[im_i]
    mask = original_masks[im_i]
    y = random.sample(range(0, im_height - patch_size_image + 1), augmented_5_per_image)
    x = random.sample(range(0, im_width - patch_size_image + 1), augmented_5_per_image)
    for i in range(augmented_5_per_image):
        angle = random.choice(list(range(-30, 31)))
        augmenter5 = iaa.Affine(rotate=angle, mode='reflect')
        image_patch = image[y[i]:y[i] + patch_size_image, x[i]:x[i] + patch_size_image]
        mask_patch = mask[y[i]:y[i] + patch_size_image, x[i]:x[i] + patch_size_image, :]
        images[counter, :, :] = augmenter5.augment_image(image_patch)
        masks[counter, :, :, :] = augmenter5.augment_image(mask_patch)
        counter += 1

print('Affine Rotate start', start)
for i in range(start, counter):
    imageio.imsave(os.path.join('D:/multiorgan/patches/images/train2', str(i) + '.png'), images[i, :, :])
    imageio.imsave(os.path.join('D:/multiorgan/patches/labels/train2', str(i) + '.png'), masks[i, :, :, :])

start = counter
print('Augmentation 6 : Affine Scale')
for im_i in tqdm(range(len(all_files))):
    image = original_images[im_i]
    mask = original_masks[im_i]
    for i in range(augmented_6_per_image):
        scale = random.choice(list(range(80, 120))) * 1.0 / 100
        new_patch_size = int(patch_size_image / scale)
        y = random.choice(range(0, im_height - new_patch_size + 1))
        x = random.choice(range(0, im_width - new_patch_size + 1))
        image_patch = image[y:y + new_patch_size, x:x + new_patch_size]
        mask_patch = mask[y:y + new_patch_size, x:x + new_patch_size, :]
        images[counter, :, :] = skimageresize(image_patch, (patch_size_image, patch_size_image), mode='constant',
                                              cval=0, preserve_range=True)
        masks[counter, :, :, :] = skimageresize(mask_patch, (patch_size_image, patch_size_image), mode='constant',
                                                cval=0, preserve_range=True)
        counter += 1

print('Affine Scale start', start)
for i in range(start, counter):
    imageio.imsave(os.path.join('D:/multiorgan/patches/images/train2', str(i) + '.png'), images[i, :, :])
    imageio.imsave(os.path.join('D:/multiorgan/patches/labels/train2', str(i) + '.png'), masks[i, :, :, :])
