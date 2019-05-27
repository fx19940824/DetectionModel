#!/usr/bin/python
# coding:utf-8

"""
@author: Yong Li
@contact: liyong@cobotsys.com
@software: PyCharm
@file: build_color.py
@time: 19-3-13 上午11:29
CAID中少螺钉判断错误的数据移动到8T机械硬盘 mobile_special_data 文件夹中
"""

import os
import cv2
import numpy as np
import shutil

mylar_mask_dir = "/media/cobot/8e505336-96a8-4c09-bc62-9ca728a68af3/new_data_dir/20190318/miss_screw"
mylar_left_image_dir = "/media/cobot/8e505336-96a8-4c09-bc62-9ca728a68af3/new_data_dir/20190318/left"
mylar_right_image_dir = "/media/cobot/8e505336-96a8-4c09-bc62-9ca728a68af3/new_data_dir/20190318/right"

mylar_ouput_left_dir = "/home/cobot/Desktop/temp/left"
mylar_ouput_right_dir = "/home/cobot/Desktop/temp/right"

for name in os.listdir(mylar_mask_dir):
    mask = cv2.imread(os.path.join(mylar_mask_dir, name), 0)
    if np.sum(mask) > 0:
        if not os.path.exists(os.path.join(mylar_ouput_left_dir, name)):
            if os.path.exists(os.path.join(mylar_left_image_dir, name)):
                print(name)
                shutil.copy(os.path.join(mylar_left_image_dir, name), os.path.join(mylar_ouput_left_dir, name))
                shutil.copy(os.path.join(mylar_right_image_dir, name), os.path.join(mylar_ouput_right_dir, name))
