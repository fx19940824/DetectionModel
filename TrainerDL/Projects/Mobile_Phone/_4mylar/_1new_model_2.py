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

# 路径与基本配置
root = os.getcwd()

# image_dir = "/media/cobot/5C8B2D882D247B56/mobile_special_data/mylar/left"
# image_dir = "/media/cobot/5C8B2D882D247B56/mobile_special_data/mylar_free/left"

# image_dir = "/home/cobot/xxxxx"
image_dir = "/home/cobot/Desktop/temp"
# image_dir = "/home/cobot/Desktop/temp"


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

    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("masks", cv2.WINDOW_GUI_NORMAL)

    for name in files:
        xxx = "y"
        print("训练b", name)
        # 读图
        image = cv2.imread(os.path.join(image_dir, name))

        # 判断B是否大于30
        index_b = image[:, :, 0] < 30

        # B/R
        image_ratio = image[:, :, 0] / (image[:, :, 2] + 1.0)
        image_ratio[image_ratio > 10] = 10
        image_ratio[index_b] = 0.0
        # if np.count_nonzero(image_ratio > ratio) > 2500:
        #     cv2.imshow("image", image)
        #     cv2.imshow("masks", (image_ratio > ratio).astype(np.uint8) * 255)
        #     cv2.waitKey()
        #     xxx = input("y or n?")
        # if xxx == "y":
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
    ratio = ratio.astype(np.float32) / 17 / 1.4

    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("mask_b", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("mid_b", cv2.WINDOW_GUI_NORMAL)

    # image_dir = "/home/cobot/Desktop/颜色异常"
    # image_dir = "/home/cobot/Pictures/full/left"
    files = os.listdir(image_dir)
    for name in files:
        print("预测", name)
        # 读图
        image = cv2.imread(os.path.join(image_dir, name))
        index_b = image[:, :, 0] > 30
        image_ratio = image[:, :, 0] / (image[:, :, 2] + 1.0)
        index_ratio = (image_ratio > ratio + 0.02) * index_b
        cv2.imshow("mid_b", index_ratio.astype(np.uint8) * 255)

        # 滤波处理
        index_ratio = cv2.blur(index_ratio.astype(np.float), (50, 50)) > 0.7

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        index_ratio = cv2.morphologyEx(index_ratio.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

        cv2.imshow("image", image)
        # cv2.imshow("mask_b", index_ratio.astype(np.uint8) * 255)
        cv2.imshow("mask_b", index_ratio)
        cv2.waitKey()

    pass


def find_bounds_r():
    files = os.listdir(image_dir)
    image = cv2.imread(os.path.join(image_dir, files[0]))

    # 构造边界线(原来有模型可用，则继续用以前的模型)
    if os.path.exists(root + "/models/ratio_r.bmp"):
        ratio_r = cv2.imread(root + "/models/ratio_r.bmp", 0)
        ratio_r = ratio_r.astype(np.float)
        ratio_r = ratio_r / 255.0
    else:
        ratio_r = np.zeros(image.shape[0:2], dtype=np.float)

    # 读图 更新 训练
    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("masks", cv2.WINDOW_GUI_NORMAL)
    for name in files:
        xxx = "y"
        print("训练r", name)
        # 读图
        image = cv2.imread(os.path.join(image_dir, name))
        image = image.astype(np.float)

        index = (image[:, :, 2] + 0.0) / (image[:, :, 0] + image[:, :, 1] + image[:, :, 2] + 0.001) > ratio_r
        # if np.count_nonzero(index) > 2500:
        #     cv2.imshow("image", image)
        #     cv2.imshow("masks", index.astype(np.uint8) * 255)
        #     cv2.waitKey()
        #     xxx = input("y or n?")
        # if xxx == "y":
        ratio_r = np.maximum(ratio_r,
                             (image[:, :, 2] + 0.0) / (image[:, :, 0] + image[:, :, 1] + image[:, :, 2] + 0.001))

    cv2.namedWindow("Ratio", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Ratio", ratio_r)
    cv2.waitKey()

    # 数据保存
    ratio_r = np.ceil(ratio_r * 255).astype(np.uint8)
    cv2.imwrite(root + "/models/ratio_r.bmp", ratio_r)
    for i in range(5):  # 按照区域保存
        cv2.imwrite(root + "/models/ratio_r_" + str(4 - i) + ".bmp", ratio_r[:, i * 1550:(i * 1550 + 1520)])


def predict_r():
    # 预测
    root = os.getcwd()
    ratio_r = cv2.imread(root + "/models/ratio_r.bmp", 0)
    ratio_r = ratio_r.astype(np.float)
    ratio_r = ratio_r / 255.0

    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("mask_r", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("mid_mask", cv2.WINDOW_GUI_NORMAL)

    files = os.listdir(image_dir)
    for name in files:
        print("预测", name)
        # 读图
        image = cv2.imread(os.path.join(image_dir, name))
        image = image.astype(np.float)
        result = ratio_r < (image[:, :, 2] + 0.0) / (image[:, :, 0] + image[:, :, 1] + image[:, :, 2] + 0.001)

        mid_res = result.copy()

        # 滤波处理
        result = cv2.blur(result.astype(np.float), (50, 50)) > 0.70
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        result = cv2.morphologyEx(result.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

        cv2.imshow("image", image / 255)
        cv2.imshow("mid_mask", mid_res.astype(np.uint8) * 255)
        cv2.imshow("mask_r", result)
        cv2.waitKey()


def main():
    find_bounds_b()
    # predict_b()
    #
    find_bounds_r()
    # predict_r()


if __name__ == "__main__":
    main()
