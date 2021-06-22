import xml.etree.ElementTree as ET
import numpy as np
import argparse
import cv2
from PIL import Image
import os
import glob
import logging
from prepare_data import create_folder

parse = argparse.ArgumentParser(description='xml annotation transfer to label img')
parse.add_argument('xml_root', default=None, metavar='XML_ROOT', type=str, help='Path to the xml dir')
parse.add_argument('save_path', default=None, metavar='LABEL_SAVE', type=str, help='Path to the label dir')


def xml2label(args):
    # if not os.path.exists(args.save_path):
    #     os.mkdir(args.save_path)
    #     print(args.save_path + "目录创建成功")
    create_folder(args.save_path)
    xml_list = glob.glob(os.path.join(args.xml_root, '*.xml'))

    for i_, xml_path in enumerate(xml_list):
        # args.xml_path = 'D:/multiorgan/Annotations/TCGA-18-5592-01Z-00-DX1.xml'
        filename = os.path.basename(xml_path).split('.')[0]
        root = ET.parse(xml_path).getroot()
        annotation_area = root.findall('./Annotation/Regions/Region')
        contours = []
        for area in annotation_area:  # 对于每一个region
            X = list(map(lambda x: float(x.get('X')),
                         area.findall('./Vertices/Vertex')))  # 取出所有X
            Y = list(map(lambda x: float(x.get('Y')),
                         area.findall('./Vertices/Vertex')))
            vertices = np.round([X, Y]).astype(int).transpose()
            contours.append(vertices)
        h = 1000
        w = 1000
        instance_label = np.zeros((h, w), dtype=np.uint16)
        for i, cont in enumerate(contours):
            cv2.fillPoly(instance_label, [cont], color=i + 1)
        instance_label_pil = Image.fromarray(instance_label)
        instance_label_pil.save(os.path.join(args.save_path, filename+'.png'))
        logging.info("{},{}/{}".format(filename, i_ + 1, len(xml_list)))
        # io.imsave('C:/Users/yzh/Desktop/test.png', instance_label)


def main():
    args = parse.parse_args()
    logging.basicConfig(level=logging.INFO)
    xml2label(args)


if __name__ == '__main__':
    main()
