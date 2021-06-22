import argparse
from PIL import Image
import numpy as np
from skimage import morphology
from prepare_data import create_folder
import os
import logging

parse = argparse.ArgumentParser(description='xml annotation transfer to label img')
parse.add_argument('labels_instance', default=None, metavar='LABEL_INSTANCE', type=str,
                   help='Path to the label_instance dir')
parse.add_argument('save_path', default=None, metavar='LABEL_SAVE', type=str, help='Path to the labels dir')


def create_ternary_labels(args):
    create_folder(args.save_path)

    image_list = os.listdir(args.labels_instance)
    for i_, img_name in enumerate(image_list):
        name = img_name.split('.')[0]
        image_path = os.path.join(args.labels_instance, img_name)

        image = np.array(Image.open(image_path))
        h, w = image.shape

        # extract edges
        id_max = np.max(image)
        contours = np.zeros((h, w), dtype=np.bool)
        nuclei_inside = np.zeros((h, w), dtype=np.bool)
        for i in range(1, id_max + 1):
            nucleus = image == i
            nuclei_inside += morphology.erosion(nucleus)
            contours += morphology.dilation(nucleus) & (~morphology.erosion(nucleus))

        ternary_label = np.zeros((h, w, 3), np.uint8)
        ternary_label[:, :, 0] = nuclei_inside.astype(np.uint8) * 255  # inside R
        ternary_label[:, :, 1] = contours.astype(np.uint8) * 255  # contours G
        ternary_label[:, :, 2] = (~(nuclei_inside + contours)).astype(np.uint8) * 255  # background B
        ternary_label_pil = Image.fromarray(ternary_label)
        ternary_label_pil.save('{:s}/{:s}_label.png'.format(args.save_path, name))
        logging.info('{},{}/{}'.format(name, i_ + 1, len(image_list)))


def main():
    args = parse.parse_args()
    logging.basicConfig(level=logging.INFO)
    create_ternary_labels(args)


if __name__ == '__main__':
    main()
