import numpy as np
from skimage.color import rgb2hed
import imageio
import argparse
import glob
import os
import sys
import logging
from PIL import Image
from skimage.exposure import rescale_intensity

def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


parse = argparse.ArgumentParser(description='generation H_stain image')
parse.add_argument('--img-dir', default=None, metavar='IMG_DIR', type=str, help='Normalize img dir')
parse.add_argument('--save-dir', default=None, metavar='SAVE_DIR', type=str, help='path to save  H_stain img')


def H_stain_generation(args):
    create_folder(args.save_dir)
    img_list = glob.glob(os.path.join(args.img_dir, '*.png'))
    for i, path in enumerate(img_list):
        name = os.path.basename(path).split('.')[0]
        # extractor = MacenkoStainExtractor
        # target = np.array(Image.open(path))
        # target = staintools.LuminosityStandardizer.standardize(target)
        # HE_matrix = extractor.get_stain_matrix(target)  # get S
        # H_matrix = np.expand_dims(HE_matrix[0], 0)
        # target_concentrations = get_concentrations(target, H_matrix)  # get C
        # OD_flat = 255 * np.exp(-1 * np.dot(target_concentrations, H_matrix))
        # H_stain = OD_flat.reshape(target.shape).astype(np.uint8)
        he_rgb = np.array(Image.open(path))
        ihc_hed = rgb2hed(he_rgb)
        h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1),
                              in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
        imageio.imsave(os.path.join(args.save_dir, name + '.png'), h)
        logging.info('name: {}, {}/{}'.format(name, i + 1, len(img_list)))


def main():
    args = parse.parse_args()
    logging.basicConfig(level=logging.INFO)
    H_stain_generation(args)


if __name__ == '__main__':
    main()
