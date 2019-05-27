#!/usr/bin/python
# coding:utf-8

"""
@author: Yong Li
@contact: liyong@cobotsys.com
@software: PyCharm
@file: build_color.py
@time: 19-1-24 上午11:29
对红蓝异常空间建模
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# image_dir = "/home/cobot/Desktop/test_color_R"
# image_dir = "/media/cobot/94584AF0584AD0A2/data/_1defect/mylar"
image_dir = "/home/cobot/Desktop/颜色异常"
# image_dir = "/home/cobot/Pictures/full/left"
files = os.listdir(image_dir)
# cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
# cv2.namedWindow("mask_hsv", cv2.WINDOW_GUI_NORMAL)
# cv2.namedWindow("bgr", cv2.WINDOW_GUI_NORMAL)
# cv2.namedWindow("mask_bgr", cv2.WINDOW_GUI_NORMAL)
for name in files:
    print(name)
    # 读图转换为HSV表达
    image = cv2.imread(os.path.join(image_dir, name))
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_hsv1 = cv2.inRange(image_hsv, (0, 0, 0), (20, 255, 255))
    mask_hsv2 = cv2.inRange(image_hsv, (160, 0, 0), (180, 255, 255))
    mask_hsv = mask_hsv1 + mask_hsv2

    # bgr = np.max(image, axis=2)[:, :, np.newaxis] - image.copy()
    # bgr_sum = np.sum(bgr, axis=2) / 255
    # bgr[:, :, 0] = bgr[:, :, 0] / bgr_sum
    # bgr[:, :, 1] = bgr[:, :, 1] / bgr_sum
    # bgr[:, :, 2] = bgr[:, :, 2] / bgr_sum
    # bgr = bgr.astype(np.uint8)
    # bgr = 255 - bgr
    # bgr_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    plt.imshow(image)
    plt.figure()
    plt.imshow(image_hsv)
    plt.show()
    # cv2.imshow("bgr", bgr)
    # mask_bgr = cv2.inRange(bgr_hsv, (100, 80, 100), (150, 255, 255))

    # cv2.imshow("image", image)
    # cv2.imshow("mask_hsv", mask_hsv)
    # # cv2.imshow("mask_bgr", ((image[:, :, 0] > 1.5 * image[:, :, 2]) * (image[:, :, 0] > 30)).astype(np.uint8) * 255)
    # cv2.waitKey()
