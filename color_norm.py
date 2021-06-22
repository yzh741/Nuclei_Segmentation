import logging
import argparse
from prepare_data import create_folder
import numpy as np
from PIL import Image
import os
import glob
# import staintools
from prepare_data import Reinhard_normalizer

parse = argparse.ArgumentParser(description='origin img color normalize')
parse.add_argument('anchor_img', default=None, metavar='ANCHOR_IMG', type=str, help='Path to the anchor image')
parse.add_argument('img_dir', default=None, metavar='IMAGE_DIR', type=str, help='Path to the original image')
parse.add_argument('save_path', default=None, metavar='COLOR_NORM_SAVE', type=str, help='Path to the color norm dir')


def color_norm(args):
    # create_folder(args.save_path)
    # dirs = os.listdir(args.img_dir)
    anchor_img = np.array(Image.open(args.anchor_img))
    normalizer = Reinhard_normalizer()
    # anchor_img = staintools.LuminosityStandardizer.standardize(anchor_img)
    # normalizer = staintools.StainNormalizer(method='macenko')
    normalizer.fit(anchor_img)
    file_list = glob.glob(args.img_dir + '/*' + '/*.png')
    for i, file in enumerate(file_list):
        # print(file)
        name_dir = os.path.dirname(file).split('\\')[-1].split('.')[0]
        create_folder(os.path.join(args.save_path, name_dir+'_norm'))
        name = os.path.basename(file).split('.')[0]  # file.split('.')[0]

        img = np.array(Image.open(file))
        # img = staintools.LuminosityStandardizer.standardize(img)
        normalized_img = normalizer.transform(img)
        normalized_img_pil = Image.fromarray(normalized_img)

        #print(os.path.join(args.save_path, name_dir+'_norm'))
        normalized_img_pil.save('{:s}/{:s}/{:s}.png'.format(args.save_path, name_dir + '_norm', name))
        logging.info('{},{}/{}'.format(name, i + 1, len(file_list)))


def main():
    args = parse.parse_args()
    logging.basicConfig(level=logging.INFO)
    color_norm(args)


if __name__ == '__main__':
    main()
