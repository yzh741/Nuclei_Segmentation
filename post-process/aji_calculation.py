import os
import glob
import numpy as np
import argparse
import logging
from PIL import Image
import sys

sys.path.append('D:/software/JetBrains/nuclei_seg')
import utils

parse = argparse.ArgumentParser(description='watershed post-processing')
parse.add_argument('img_dir', default=None, metavar='IMG_DIR', type=str, help='path to the img')
parse.add_argument('label_dir', default=None, metavar='LABEL_DIR', type=str, help='path to the ground truth')
parse.add_argument('save_dir', default=None, metavar='SAVE_DIR', type=str, help='Path to save the result')


def cal_aji(args):
    avg_results = utils.AverageMeter(8)
    all_results = dict()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    imgs = glob.glob(os.path.join(args.img_dir, '*.tiff'))
    labels = glob.glob(os.path.join(args.label_dir, '*.png'))
    for i, path in enumerate(labels):
        gt = np.array(Image.open(path))
        gt_name = os.path.basename(path).split('.')[0]
        pred = np.array(Image.open(imgs[i]))
        pred_name = os.path.basename(imgs[i]).split('.')[0].split('_')[:-1]
        pred_name = '_'.join(pred_name)
        print(pred_name, gt_name)
        assert pred_name == gt_name
        result_object = utils.nuclei_accuracy_object_level(pred, gt)
        result = utils.accuracy_pixel_level(np.expand_dims(pred > 0, 0), np.expand_dims(gt > 0, 0))
        pixel_accu = result[0]
        all_results[gt_name] = tuple([pixel_accu, *result_object])
        avg_results.update([pixel_accu, *result_object])
    strs = args.img_dir.split('//')[-1].split('_')[0]
    print(strs)
    header = ['pixel_acc', 'recall', 'precision', 'F1', 'Dice', 'IoU', 'Hausdorff', 'AJI']
    save_results(header, avg_results.avg, all_results, '{:s}/{:s}_result.txt'.format(args.save_dir, strs[-1]))


def main():
    args = parse.parse_args()
    logging.basicConfig(level=logging.INFO)
    cal_aji(args)


def save_results(header, avg_results, all_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    assert N == len(avg_results)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(avg_results[i]))
        file.write('{:.4f}\n'.format(avg_results[N - 1]))
        file.write('\n')

        # all results
        for key, values in sorted(all_results.items()):
            file.write('{:s}:'.format(key))
            for value in values:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')


if __name__ == '__main__':
    main()
