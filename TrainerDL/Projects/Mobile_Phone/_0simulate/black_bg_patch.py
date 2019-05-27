#!/usr/bin/python  
# coding:utf-8  

""" 
@author: Yong Li 
@contact: liyong@cobotsys.com
@software: PyCharm 
@file: black_bg_patch.py 
@time: 18-11-21 下午2:42 
"""
import cv2
import os
import numpy as np

img_dir = "/media/yong/data/cellphone_project/data/screw/raw_data/patches/case1"
out_dir = "/media/yong/data/cellphone_project/data/screw/raw_data/patches/temp"
files = os.listdir(img_dir)

for short_name in files:
    img = cv2.imread(os.path.join(img_dir, short_name))
    img[img == 255] = 0
    cv2.imwrite(os.path.join(out_dir, short_name), img)


