#!/usr/bin/python
# coding:utf-8

"""
@author: Yong Li
@contact: liyong@cobotsys.com
@software: PyCharm
@file: syn_images.py
@time: 2019年02月28日11:16:37
删除没有配对的图像
"""

# import cv2
# import numpy as np
import os
# import random
# import copy
import shutil
from ctools.basic_func import get_all_files

# import matplotlib.pyplot as plt
#
# left_image_dir = "/home/cobot/Pictures/full/left"
# right_image_dir = "/home/cobot/Pictures/full/right"
# out_dir = "/home/cobot/Pictures/full/bad"

left_image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/20190312/left"
right_image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/20190312/right"
out_dir = "/home/cobot/Desktop/多余的"

left_names = get_all_files(left_image_dir)
right_names = get_all_files(right_image_dir)

for left_name in left_names:
    if left_name not in right_names:
        if not os.path.exists(os.path.dirname(os.path.join(out_dir, left_name))):
            os.makedirs(os.path.dirname(os.path.join(out_dir, left_name)))
        shutil.move(os.path.join(left_image_dir, left_name), os.path.join(out_dir, left_name))

for right_name in right_names:
    if right_name not in left_names:
        if not os.path.exists(os.path.dirname(os.path.join(out_dir, right_name))):
            os.makedirs(os.path.dirname(os.path.join(out_dir, right_name)))
        shutil.move(os.path.join(right_image_dir, right_name), os.path.join(out_dir, right_name))

# temp_image = cv2.imread("/home/cobot/Pictures/full/left/2019.02.27_17:06:49.437_.bmp", 0)
# temp_image = 0.0 + temp_image - np.mean(temp_image[:])
# temp_image = temp_image / np.sqrt(np.sum(temp_image[:] * temp_image[:]))
