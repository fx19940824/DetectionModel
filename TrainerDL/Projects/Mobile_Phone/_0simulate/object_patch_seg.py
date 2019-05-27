#!/usr/bin/python  
# coding:utf-8  

""" 
@author: Yong Li 
@contact: liyong@cobotsys.com
@software: PyCharm 
@file: image_seg.py 
@time: 18-10-10 下午1:49
螺钉图像分割.从纯色背景上截取螺钉的ROI区域并存储.
输入:系列螺钉图像,纯色背景
输出:含螺钉的图像片,mask使用255表示
version:1.0
修改时间:2018年11月17日
"""
import cv2
import numpy as np
import os
from skimage.measure import label, regionprops

# ---------------主程序部分-------------------------------------------------------------------------------------------
root = ""
image_dir = "/media/yong/data/cellphone_project/test_data"  # 原始纯色背景螺钉图像的目录
output_dir = "/media/yong/data/cellphone_project/temp"  # 输出位置
files = os.listdir(image_dir)

new_ind = 0  # 新的mask的计数
for i, path in enumerate(files):
    image = cv2.imread(os.path.join(image_dir, path))  # 读图
    mask = cv2.inRange(image, (0, 0, 0), (254, 254, 254))  # 获取mask,也就是值为255的像素

    # 使用mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_image[masked_image == 0] = 255

    label_image = label(mask)
    props = regionprops(label_image)

    # 根据mask的blob的裁剪图像
    for k in range(len(props)):
        center_points = np.array(props[k].centroid).astype(np.int)
        if center_points[0] in range(54, masked_image.shape[0] - 54) and center_points[1] in range(54,
                                                                                                   masked_image.shape[
                                                                                                       1] - 54):
            image_patch = masked_image[(center_points[0] - 53):center_points[0] + 53,
                          center_points[1] - 53:center_points[1] + 53, :]

            new_ind = new_ind + 1
            cv2.imwrite(output_dir + os.sep + "xxx_" + str(new_ind) + ".png", image_patch)
