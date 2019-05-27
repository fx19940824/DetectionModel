#!/usr/bin/python
# coding:utf-8

"""
@author: Yong Li
@contact: liyong@cobotsys.com
@software: PyCharm
@file: build_color.py
@time: 19-1-24 上午11:29
使用颜色对采集的数据进行滤出和分类
"""

import os
import cv2
import numpy as np
import shutil

import matplotlib.pyplot as plt

image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/20190228/left"
output_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/fixed_phones/left"
files = os.listdir(image_dir)

root = os.getcwd()

# Step.1 加载模型
ratio = cv2.imread(root + "/models/ratio.bmp", 0)
ratio = ratio.astype(np.float32) / 17

h_min = cv2.imread(root + "/models/h_min_r.bmp", 0)
h_max = cv2.imread(root + "/models/h_max_r.bmp", 0)

# cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
# cv2.namedWindow("mask_b", cv2.WINDOW_GUI_NORMAL)
# cv2.namedWindow("mask_r", cv2.WINDOW_GUI_NORMAL)
for name in files:
    print("预测", name)
    # 读图
    image = cv2.imread(os.path.join(image_dir, name))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]

    # 蓝色薄膜判断
    index_b = image[:, :, 0] > 30
    image_ratio = image[:, :, 0] / (image[:, :, 2] + 1.0)
    index_ratio = (image_ratio > ratio) * index_b
    # 滤波处理
    result_b = cv2.blur(index_ratio.astype(np.float), (150, 150)) > 0.65
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    result_b = cv2.morphologyEx(result_b.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

    # 红色薄膜判断
    result = ((h > h_max) + (h < h_min)) * (image[:, :, 2] > image[:, :, 0]) * (image[:, :, 2] > image[:, :, 1])
    mid_res = result.copy()
    # 滤波处理
    result_r = cv2.blur(result.astype(np.float), (150, 150)) > 0.70
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    result_r = cv2.morphologyEx(result_r.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

    if np.sum(result_b[:] + result_r[:]) > 255*100:
        # cv2.imshow("image", image)
        # cv2.imshow("mask_b", result_b)
        # cv2.imshow("mask_r", result_r)
        # cv2.waitKey()
        shutil.move(os.path.join(image_dir, name), os.path.join(output_dir, name))


pass


# def predict_b():
#     # 预测
#     ratio = cv2.imread("./models/ratio.bmp", 0)
#     ratio = ratio.astype(np.float32) / 17
#
#     cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
#     cv2.namedWindow("mask", cv2.WINDOW_GUI_NORMAL)
#
#     image_dir = "/home/cobot/Desktop/test_color"
#     # image_dir = "/home/cobot/Pictures/full/left"
#     files = os.listdir(image_dir)
#     for name in files:
#         print("预测", name)
#         # 读图
#         image = cv2.imread(os.path.join(image_dir, name))
#         index_b = image[:, :, 0] > 30
#         image_ratio = image[:, :, 0] / (image[:, :, 2] + 1.0)
#         index_ratio = (image_ratio > ratio) * index_b
#
#         # 滤波处理
#         index_ratio = cv2.blur(index_ratio.astype(np.float), (75, 75)) > 0.65
#
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#         index_ratio = cv2.morphologyEx(index_ratio.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
#
#         cv2.imshow("image", image)
#         # cv2.imshow("mask", index_ratio.astype(np.uint8) * 255)
#         cv2.imshow("mask", index_ratio)
#         cv2.waitKey()
#     pass
#
#
# def predict_r():
#     # 预测
#     root = os.getcwd()
#     h_min = cv2.imread(root + "/models/h_min_r.bmp", 0)
#     h_max = cv2.imread(root + "/models/h_max_r.bmp", 0)
#
#     cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
#     cv2.namedWindow("mask", cv2.WINDOW_GUI_NORMAL)
#     cv2.namedWindow("mid_mask", cv2.WINDOW_GUI_NORMAL)
#
#     image_dir = "/home/cobot/Desktop/test_color_R"
#     # image_dir = "/home/cobot/Pictures/full/left"
#     files = os.listdir(image_dir)
#     for name in files:
#         print("预测", name)
#         # 读图
#         image = cv2.imread(os.path.join(image_dir, name))
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         h = hsv[:, :, 0]
#
#         result = ((h > h_max) + (h < h_min)) * (image[:, :, 2] > image[:, :, 0]) * (image[:, :, 2] > image[:, :, 1])
#         mid_res = result.copy()
#
#         # 滤波处理
#         result = cv2.blur(result.astype(np.float), (50, 50)) > 0.70
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#         result = cv2.morphologyEx(result.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
#
#         cv2.imshow("image", image)
#         # cv2.imshow("mask", result.astype(np.uint8) * 255)
#         cv2.imshow("mask", result)
#         cv2.imshow("mid_mask", mid_res.astype(np.uint8) * 255)
#         cv2.waitKey()
#     pass
#
#
# def main():
#     # find_bounds_b()
#     # predict_b()
#
#     # find_bounds_r()
#     predict_r()
#
#
# if __name__ == "__main__":
#     main()
