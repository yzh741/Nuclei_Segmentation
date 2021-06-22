import numpy as np
import cv2
import random
import os
import json
import glob
import imageio

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""

# train_txt_path = os.path.join("..", "..", "Data/train.txt")

CNum = 1000  # 挑选多少图片进行计算

img_h, img_w = 250, 250
imgs = np.zeros([img_w, img_h, 4, 1])
means, stdevs = [], []
train_imgs = glob.glob('D:/multiorgan/patches//images/train/*.png')
random.shuffle(train_imgs)
# train_imgs = [os.path.basename(path) for path in train_imgs]

count = 0
for i, img_path in enumerate(train_imgs):
    # img_path = os.path.join('D:/oct/imgs', name)

    img = imageio.imread(img_path)
    # img = cv2.resize(img, (img_h, img_w))
    img = img[:, :, :, np.newaxis]
    imgs = np.concatenate((imgs, img), axis=3)
    print('{}/{}'.format(i + 1, len(train_imgs)))
    count += 1
    if count > 1000:
        break

print(imgs.shape)
imgs = imgs.astype(np.float32) / 255.

for i in range(4):
    pixels = imgs[:, :, i, 1:].flatten()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# means.reverse()  # BGR --> RGB  RGBA
# stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
