import os
import glob
import random
import argparse
import imageio
import logging
import numpy as np

parse = argparse.ArgumentParser(description='watershed post-processing')
parse.add_argument('img_dir', default=None, metavar='IMG_DIR', type=str, help='path to the img')
parse.add_argument('save_dir', default=None, metavar='SAVE_DIR', type=str, help='Path to save the result')


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b


def visual(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    imgs = glob.glob(os.path.join(args.img_dir, '*.tiff'))
    for i, path in enumerate(imgs):
        name = os.path.basename(path).split('.')[0]
        img = np.array(imageio.imread(path))
        h, w = img.shape
        colored = np.zeros((h, w, 3))
        for k in range(1, img.max() + 1):
            colored[img == k, :] = np.array(get_random_color())
        colored = (colored * 255).astype(np.uint8)
        imageio.imsave(os.path.join(args.save_dir, name + '.png'), colored)
        logging.info('name: {}, {}/{}'.format(name, i+1, len(imgs)))


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse.parse_args()
    visual(args)


if __name__ == '__main__':
    main()
