# Nuclei_Segmentation
This repository contains the pytorch code for the paper:  Improving Nuclei/Gland Instance Segmentation in Histopathology Images by Full Resolution Neural Network and Spatial Constrained Loss, MICCAI2019. ([pdf](https://link.springer.com/chapter/10.1007%2F978-3-030-32239-7_42))  

The code's original github([link](https://github.com/huiqu18/FullNet-varCE))  

### Data description
Before training, you need to prepare the training images and labels. The Data download ([link](https://nucleisegmentationbenchmark.weebly.com/dataset.html))
Note. The authors of the Multi-Organ dataset replaced some images in the dataset in September 2019. Therefore the current dataset is different from what I used for experiments. As a result, we put the original dataset ([here](https://drive.google.com/drive/folders/19ASx6rRvvTapp_W80oGLxNwoN8eLKTil?usp=sharing)) (Tissue Images and Annotations) for your reference. The color normalized images、instance labels and ternary labels are also included in the folder. 
| Image | Ternary label |
| ---|-----|
| ![](images/TCGA-18-5592-01Z-00-DX1_0.png) | ![](images/TCGA-18-5592-01Z-00-DX1_0_label.png) | 

## Usage

* arrange data like what my google drive did

* For MultiOrgan dataset, generate the instance labels from the .xml annotation files. Use `xml2label.py`.

* Generate the ternary label from each instance label using `ternary_img_gen`.

* if you want to use weight_map,Compute the weight map from each instance label using `weight_map.m`.

* For MultiOrgan dataset, perform color normalization in all images to reduce the color variance using `color_norm.py` .

* For MultiOrgan dataset, split each large training image/label/weight_map into 16 small 250x250 patches 
using `split_patches.py`, you only need to do this for your train_set 

### Model training and testing
To training a model, set related parameters in the file `options.p` and run the script sh `scripts/train.sh`
run `scripts/test_same.sh` or `scripts/test_diff.sh` to evaluate the two test sets
