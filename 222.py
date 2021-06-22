import torch
import glob
import random
import numpy as np
from models.layers import AMSAF_block, RRAMSAF_block
from models.r2_unet import R2U_NetV2
from torchvision.models import AlexNet
import pandas as pd
from scipy import stats

frame = pd.read_excel('C:/Users/yzh/Desktop/qulication.xlsx', sheet_name=None)
frmae2 = pd.read_excel('C:/users/yzh/Desktop/8.xlsx', sheet_name=None)
category3 = frame['category3']
category1 = frame['category1']
category7 = frame['category7']
category4 = frame['category4']
category5 = frame['category5']
category734 = frame['734']
category125 = frame['125']
category8_1 = frmae2['8-1']
category8_2 = frmae2['8-2']

pair_71 = {}
pair_45 = {}
pair_734_125 = {}
pair_3_8_1 = {}
pair_8_1_8_2 = {}
pair_734_1258_1 = {}
pair_125_8_1 = {}
for col_name in category125.columns:
    sum=0
    data1 = category125[col_name].values.tolist()
    data1 = random.sample(data1, 10)
    data3 = category8_1[col_name].values.tolist()
    for i in range(10):
        _, p_value = stats.ttest_rel(data1, data3)
        sum += p_value
    pair_125_8_1[col_name] = sum/10

print('pair_125_8_1:', pair_125_8_1)
# pair_23 = {}
# for col_name in category1.columns:
#     data1 = category7[col_name].values
#     data2 = category1[col_name].values
#     _, p_value = stats.ttest_rel(data1, data2)
#     pair_71[col_name] = p_value
#
# print('pair_71:', pair_71)
#
# for col_name in category1.columns:
#     data1 = category4[col_name].values
#     data2 = category5[col_name].values
#     _, p_value = stats.ttest_rel(data1, data2)
#     pair_45[col_name] = p_value
#
# print('pair_45:', pair_45)
#
# for col_name in category734.columns:
#     data1 = category734[col_name].values
#     data2 = category125[col_name].values
#     _, p_value = stats.ttest_rel(data1, data2)
#     pair_734_125[col_name] = p_value
#
# print('pair_734_125:', pair_734_125)


