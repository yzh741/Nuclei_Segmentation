#!/bin/bash
#source ~/.bashrc
#hostname

python /home/yzh/nuclei_segmentation/nuclei_seg/test.py \
    --model-path /home/yzh/nuclei_segmentation/experiment/result/checkpoints/checkpoint_best.pth.tar \
    --img-dir /home/yzh/nuclei_segmentation/data/multiorgan/images/test_diff \
    --label-dir /home/yzh/nuclei_segmentation/data/multiorgan/labels_instance/test_diff \
    --save-dir /home/yzh/nuclei_segmentation/experiment/result/test_result
