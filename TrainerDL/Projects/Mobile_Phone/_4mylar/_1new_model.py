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

root = os.getcwd()

image_dir = "/media/cobot/5C8B2D882D247B56/mobile_special_data/mylar/left"


def find_bounds_b():
    # 路径与基本配置
    # image_dir = "/media/cobot/94584AF0584AD0A2/data/normal_images/20190228/left"
    # image_dir = "/home/cobot/Desktop/left"
    files = os.listdir(image_dir)
    image = cv2.imread(os.path.join(image_dir, files[0]))

    # 构造红蓝边界线(原来有模型可用，则继续用以前的模型)
    root = os.getcwd()

    # 构造红蓝边界线(原来有模型可用，则继续用以前的模型)
    if os.path.exists(root + "/models/ratio.bmp"):
        ratio = cv2.imread(root + "/models/ratio.bmp", 0).astype(np.float) / (1.5 * 17)
    else:
        ratio = np.zeros(image.shape[0:2]) + 0.0

    for name in files:
        print("训练b", name)
        # 读图
        image = cv2.imread(os.path.join(image_dir, name))

        # 判断B是否大于30
        index_b = image[:, :, 0] < 30

        # B/R
        image_ratio = image[:, :, 0] / (image[:, :, 2] + 1.0)
        image_ratio[image_ratio > 10] = 10
        image_ratio[index_b] = 0.0

        ratio = np.maximum(ratio, image_ratio)

    ratio = ratio * 1.5
    ratio_out = (ratio * 17).astype(np.uint8)
    # 数据保存
    cv2.imwrite(root + "/models/ratio.bmp", ratio_out)
    for i in range(5):  # 按照区域保存
        cv2.imwrite(root + "/models/ratio_" + str(4 - i) + ".bmp", ratio_out[:, i * 1550:(i * 1550 + 1520)])


def predict_b():
    # 预测
    ratio = cv2.imread(root + "/models/ratio.bmp", 0)
    ratio = ratio.astype(np.float32) / 17

    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_GUI_NORMAL)

    # image_dir = "/home/cobot/Desktop/颜色异常"
    # image_dir = "/home/cobot/Pictures/full/left"
    files = os.listdir(image_dir)
    for name in files:
        print("预测", name)
        # 读图
        image = cv2.imread(os.path.join(image_dir, name))
        index_b = image[:, :, 0] > 30
        image_ratio = image[:, :, 0] / (image[:, :, 2] + 1.0)
        index_ratio = (image_ratio > ratio) * index_b

        # 滤波处理
        index_ratio = cv2.blur(index_ratio.astype(np.float), (75, 75)) > 0.65

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        index_ratio = cv2.morphologyEx(index_ratio.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

        cv2.imshow("image", image)
        # cv2.imshow("mask", index_ratio.astype(np.uint8) * 255)
        cv2.imshow("mask", index_ratio)
        cv2.waitKey()

    pass


def find_bounds_r():
    # 路径与基本配置
    # image_dir = "/media/cobot/94584AF0584AD0A2/data/normal_images/20190228/left"
    # image_dir = "/home/cobot/Desktop/left"
    # image_dir = "/home/cobot/Desktop/颜色异常"
    files = os.listdir(image_dir)
    image = cv2.imread(os.path.join(image_dir, files[0]))

    # 构造红蓝边界线(原来有模型可用，则继续用以前的模型)
    if os.path.exists(root + "/models/h_min_r.bmp"):
        h_min = cv2.imread(root + "/models/h_min_r.bmp", 0) + 2
        h_max = cv2.imread(root + "/models/h_max_r.bmp", 0) - 2
    else:
        h_min = np.zeros(image.shape[0:2]) + 30
        h_max = np.zeros(image.shape[0:2]) + 150

    for name in files:
        print("训练r", name)
        # 读图
        image = cv2.imread(os.path.join(image_dir, name))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]

        # 判断R是否大于B和G
        index_r = (h < 20) + (h > 160)
        index_r = (image[:, :, 2] <= image[:, :, 0]) + (image[:, :, 2] <= image[:, :, 1]) + (hsv[:, :, 1] < 100)
        h[index_r] = 90

        # 更新边界
        h_min = np.minimum(h_min, h)
        h_max = np.maximum(h_max, h)

    h_min = h_min - 2  # 适度缩放范围
    h_max = h_max + 2
    np.clip(h_min, 0, 255, h_min)
    np.clip(h_max, 0, 255, h_max)

    # 数据保存
    cv2.imwrite(root + "/models/h_min_r.bmp", h_min)
    cv2.imwrite(root + "/models/h_max_r.bmp", h_max)
    for i in range(5):  # 按照区域保存
        cv2.imwrite(root + "/models/h_min_r_" + str(4 - i) + ".bmp", h_min[:, i * 1550:(i * 1550 + 1520)])
        cv2.imwrite(root + "/models/h_max_r_" + str(4 - i) + ".bmp", h_max[:, i * 1550:(i * 1550 + 1520)])


def predict_r():
    # 预测
    root = os.getcwd()
    h_min = cv2.imread(root + "/models/h_min_r.bmp", 0)
    h_max = cv2.imread(root + "/models/h_max_r.bmp", 0)

    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("mid_mask", cv2.WINDOW_GUI_NORMAL)

    # image_dir = "/home/cobot/Desktop/颜色异常"
    # image_dir = "/home/cobot/Pictures/full/left"
    files = os.listdir(image_dir)
    for name in files:
        print("预测", name)
        # 读图
        image = cv2.imread(os.path.join(image_dir, name))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]

        result = (s >= 100) * ((h > h_max) + (h < h_min)) * (image[:, :, 2] > image[:, :, 0]) * (
            image[:, :, 2] > image[:, :, 1])
        mid_res = result.copy()

        # 滤波处理
        result = cv2.blur(result.astype(np.float), (50, 50)) > 0.70
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        result = cv2.morphologyEx(result.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

        cv2.imshow("image", image)
        # cv2.imshow("mask", result.astype(np.uint8) * 255)
        cv2.imshow("mask", result)
        cv2.imshow("mid_mask", mid_res.astype(np.uint8) * 255)
        cv2.waitKey()
        # plt.imshow(image)
        # plt.show()
    pass


def main():
    # find_bounds_b()
    # predict_b()
    #
    find_bounds_r()
    predict_r()


if __name__ == "__main__":
    main()
