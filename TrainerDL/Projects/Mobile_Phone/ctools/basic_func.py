#!/usr/bin/python  
# coding:utf-8  

""" 
@author: Yong Li 
@contact: liyong@cobotsys.com
@software: PyCharm 
@file: basic_func.py 
@time: 18-8-7 下午7:19
1. Cell phone图像与模板图像对齐
2. 清空文件夹下所有的文件
"""

import numpy as np
import cv2
import os
# from skimage.measure import label, regionprops

import random
import shutil
# import lmdb
import sys


# ---获取路径下所有文件函数------------------------------------------------------------------------------------------------
def get_all_files(root):
    output = []
    for roots, dir, files in os.walk(root, followlinks=True):
        for short_name in files:
            output = output + [os.path.join(roots, short_name)[len(root) + 1:]]
            # for dir_name in dir:
            #     output = output + get_all_files(os.path.join(roots, dir_name))
    return output


# ---测试get_all_files函数-----------------------------------------------------------------------------------------------
def test_get_all_files():
    root = "/media/yong/data/test/yong"
    print(get_all_files(root))


# 清空文件函数
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


if __name__ == "__main__":
    pass
    # test_image_correct()
    # test_get_all_files()
