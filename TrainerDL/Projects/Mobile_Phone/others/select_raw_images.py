#!/usr/bin/python
# coding:utf-8

"""
@author: Yong Li
@contact: liyong@cobotsys.com
@software: PyCharm
@file: syn_images.py
@time: 2019年02月28日11:16:37
删除采集到的不好的图像
直接利用点乘相似性计算
"""

import cv2
import numpy as np
import os
import random
import copy
import time
import shutil
import matplotlib.pyplot as plt
from ctools.basic_func import get_all_files

# image_dir = "/home/cobot/Pictures/full/left"
# out_dir = "/home/cobot/Pictures/full/bad"
# image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/left"
image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/20190312/right"
out_dir = "/home/cobot/Desktop/多余的"
# temp_image = cv2.imread("/home/cobot/Pictures/full/left/2019.02.27_17:06:49.437_.bmp", 0)
# temp_image = cv2.imread("/home/cobot/Desktop/left/2019.03.05_20.10.34.306_.bmp", 0)
temp_image = cv2.imread("/media/cobot/94584AF0584AD0A2/data/_0normal_images/templates.bmp", 0)
temp_image = 0.0 + temp_image - np.mean(temp_image[:])
temp_image = temp_image / np.sqrt(np.sum(temp_image[:] * temp_image[:]))

# names = os.listdir(image_dir)
names = get_all_files(image_dir)#[3500:]

start_time = time.time()
for k, name in enumerate(names):
    image = cv2.imread(os.path.join(image_dir, name), 0)
    if image is None:
        continue
    image = 0.0 + image - np.mean(image[:])

    cc = np.sum(image[:] * temp_image[:]) / np.sqrt(np.sum(image[:] * image[:]))

    print(name, cc)
    if cc < 0.8:  # 0.7
        if not os.path.exists(os.path.dirname(os.path.join(out_dir, name))):
            os.makedirs(os.path.dirname(os.path.join(out_dir, name)))
        shutil.move(os.path.join(image_dir, name), os.path.join(out_dir, name))
        # plt.figure()
        # plt.imshow(image)
        # plt.show()
    if k % 10 == 0:
        end_time = time.time()
        print(str(k) + ":" + str(end_time - start_time))
