import argparse
import numpy as np
import glob
import os
import logging
from PIL import Image
import imageio
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

parse = argparse.ArgumentParser(description='watershed post-processing')
parse.add_argument('predict_dir', default=None, metavar='PREDICT_DIR', type=str, help='path to the model predict')
parse.add_argument('save_dir', default=None, metavar='PREDICT_DIR', type=str, help='Path to save the result')


def watershed_process(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    masks = glob.glob(os.path.join(args.predict_dir, '*.tiff'))


    for i, path in enumerate(masks):
        name = os.path.basename(path).split('.')[0].split('_')
        name = '_'.join(name[:3])
        mask = np.array(Image.open(path))
        distance = ndi.distance_transform_edt(mask)
        local_maxi = peak_local_max(distance, labels=(mask > 0.6), min_distance=10, indices=False)
        markers = ndi.label(local_maxi)[0]
        image = watershed(-distance, markers, mask=mask)
        imageio.imsave(os.path.join(args.save_dir, name + '.tiff'), image)
        logging.info('name: {}, {}/{}'.format(name, i + 1, len(masks)))


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse.parse_args()
    watershed_process(args)


if __name__ == '__main__':
    main()
