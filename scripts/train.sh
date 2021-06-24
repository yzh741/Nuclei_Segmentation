#!/bin/bash
#source ~/.bashrc
#hostname

python /home/yzh/nuclei_segmentation/nuclei_seg/train.py \
    --alpha 0.0 \
    --batch-size 8 \
    --lr 0.001 \
    --epochs 100 \
    --data-dir /home/yzh/nuclei_segmentation/data/multiorgan \
    --save-dir /home/yzh/nuclei_segmentation/experiment/result
