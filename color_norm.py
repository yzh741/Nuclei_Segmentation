import logging
import argparse
from prepare_data import create_folder, Reinhard_normalizer
import numpy as np
from PIL import Image
import os

parse = argparse.ArgumentParser(description='origin img color normalize')
parse.add_argument('anchor_img', default=None, metavar='ANCHOR_IMG', type=str, help='Path to the anchor image')
parse.add_argument('img_dir', default=None, metavar='IMAGE_DIR', type=str, help='Path to the original image')
parse.add_argument('save_path', default=None, metavar='COLOR_NORM_SAVE', type=str, help='Path to the color norm dir')


def color_norm(args):
    create_folder(args.save_path)
    anchor_img = np.array(Image.open(args.anchor_img))
    normalizer = Reinhard_normalizer()
    normalizer.fit(anchor_img)
    file_list = os.listdir(args.img_dir)
    for i, file in enumerate(file_list):
        name = file.split('.')[0]
        img = np.array(Image.open('{:s}/{:s}'.format(args.img_dir, file)))
        normalized_img = normalizer.transform(img)
        normalized_img_pil = Image.fromarray(normalized_img)
        normalized_img_pil.save('{:s}/{:s}.png'.format(args.save_path, name))
        logging.info('{},{}/{}'.format(name, i + 1, len(file_list)))


def main():
    args = parse.parse_args()
    logging.basicConfig(level=logging.INFO)
    color_norm(args)


if __name__ == '__main__':
    main()
