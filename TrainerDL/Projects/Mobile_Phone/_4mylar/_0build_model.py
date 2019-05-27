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


def find_bounds():
    # 路径与基本配置
    # image_dir = "/media/cobot/94584AF0584AD0A2/data/normal_images/20190228/left"
    image_dir = "/home/cobot/Desktop/left"
    files = os.listdir(image_dir)[0:2]
    image = cv2.imread(os.path.join(image_dir, files[0]))

    # 构造红蓝边界线(原来有模型可用，则继续用以前的模型)
    root = os.getcwd()
    h_max_b = cv2.imread(root + "/models/h_max_b.bmp", cv2.IMREAD_GRAYSCALE)
    h_min_b = cv2.imread(root + "/models/h_min_b.bmp", cv2.IMREAD_GRAYSCALE)
    s_max_b = cv2.imread(root + "/models/s_max_b.bmp", cv2.IMREAD_GRAYSCALE)
    s_min_b = cv2.imread(root + "/models/s_min_b.bmp", cv2.IMREAD_GRAYSCALE)
    v_max_b = cv2.imread(root + "/models/v_max_b.bmp", cv2.IMREAD_GRAYSCALE)
    v_min_b = cv2.imread(root + "/models/v_min_b.bmp", cv2.IMREAD_GRAYSCALE)
    h_max_r = cv2.imread(root + "/models/h_max_r.bmp", cv2.IMREAD_GRAYSCALE)
    h_min_r = cv2.imread(root + "/models/h_min_r.bmp", cv2.IMREAD_GRAYSCALE)
    s_max_r = cv2.imread(root + "/models/s_max_r.bmp", cv2.IMREAD_GRAYSCALE)
    s_min_r = cv2.imread(root + "/models/s_min_r.bmp", cv2.IMREAD_GRAYSCALE)
    v_max_r = cv2.imread(root + "/models/v_max_r.bmp", cv2.IMREAD_GRAYSCALE)
    v_min_r = cv2.imread(root + "/models/v_min_r.bmp", cv2.IMREAD_GRAYSCALE)

    if h_max_b is None or s_max_b is None or v_max_b is None or \
                    h_min_b is None or s_min_b is None or v_min_b is None or \
                    h_max_r is None or s_max_r is None or v_max_r is None or \
                    h_min_r is None or s_min_r is None or v_min_r is None or True:
        h_max_b = np.zeros(image.shape[0:2]) + 150
        h_min_b = np.zeros(image.shape[0:2]) + 90
        s_max_b = np.zeros(image.shape[0:2]) + 255
        s_min_b = np.zeros(image.shape[0:2]) + 20
        v_max_b = np.zeros(image.shape[0:2]) + 255
        v_min_b = np.zeros(image.shape[0:2]) + 40

        h_max_r = np.zeros(image.shape[0:2]) + 40
        s_max_r = np.zeros(image.shape[0:2]) + 255
        v_max_r = np.zeros(image.shape[0:2]) + 255
        h_min_r = np.zeros(image.shape[0:2]) + 160
        s_min_r = np.zeros(image.shape[0:2]) + 20
        v_min_r = np.zeros(image.shape[0:2]) + 40

    for name in files:
        # # # Debug显示
        # plt.figure(0)
        # plt.hold(True)
        # plt.plot(h_max_b[:, 600], 'r')
        # plt.plot(h_min_b[:, 600], 'b')

        # plt.figure(1)
        # plt.plot(s_max_b[:, 600], 'r')
        # plt.plot(s_min_b[:, 600], 'b')
        # plt.figure(2)
        # plt.plot(v_max_b[:, 600], 'r')
        # plt.plot(v_min_b[:, 600], 'b')
        print(name)
        # 读图转换为HSV表达
        image = cv2.imread(os.path.join(image_dir, name))
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = image_hsv[:, :, 0]
        s = image_hsv[:, :, 1]
        v = image_hsv[:, :, 2]

        # 判断颜色值是否位于缺陷区域内
        index_b = h < h_max_b
        index_b = index_b * (h > h_min_b)
        index_b = index_b * (s < s_max_b)
        index_b = index_b * (s > s_min_b)
        index_b = index_b * (v < v_max_b)
        index_b = index_b * (v > v_min_b)

        index_r = h < h_max_r
        index_r = index_r + (h > h_min_r)
        index_r = index_r * (s < s_max_r)
        index_r = index_r * (s > s_min_r)
        index_r = index_r * (v < v_max_r)
        index_r = index_r * (v > v_min_r)

        # 计算6*2个距离
        dist_h_max_b = (h_max_b - h) * (s_max_b - s_min_b) * (v_max_b - v_min_b)
        dist_h_min_b = (-h_min_b + h) * (s_max_b - s_min_b) * (v_max_b - v_min_b)
        dist_s_max_b = (s_max_b - s) * (h_max_b - h_min_b) * (v_max_b - v_min_b) + 100000
        dist_s_min_b = (-s_min_b + s) * (h_max_b - h_min_b) * (v_max_b - v_min_b)
        dist_v_max_b = (v_max_b - v) * (h_max_b - h_min_b) * (s_max_b - s_min_b) + 100000
        dist_v_min_b = (-v_min_b + v) * (h_max_b - h_min_b) * (s_max_b - s_min_b)
        dist_b = np.concatenate((dist_h_max_b[:, :, np.newaxis], dist_h_min_b[:, :, np.newaxis],
                                 dist_s_max_b[:, :, np.newaxis], dist_s_min_b[:, :, np.newaxis],
                                 dist_v_max_b[:, :, np.newaxis], dist_v_min_b[:, :, np.newaxis]), axis=2)

        dist_h_max_r = (h_max_r - image_hsv[:, :, 0]) * (s_max_r - s_min_r) * (v_max_r - v_min_r)
        dist_h_min_r = (-h_min_r + image_hsv[:, :, 0]) * (s_max_r - s_min_r) * (v_max_r - v_min_r)
        dist_s_max_r = (s_max_r - image_hsv[:, :, 1]) * (h_max_r + 180 - h_min_r) * (v_max_r - v_min_r) + 100000
        dist_s_min_r = (-s_min_r + image_hsv[:, :, 1]) * (h_max_r + 180 - h_min_r) * (v_max_r - v_min_r)
        dist_v_max_r = (v_max_r - image_hsv[:, :, 2]) * (h_max_r + 180 - h_min_r) * (s_max_r - s_min_r) + 100000
        dist_v_min_r = (-v_min_r + image_hsv[:, :, 2]) * (h_max_r + 180 - h_min_r) * (s_max_r - s_min_r)
        dist_h_max_r[dist_h_max_r < 0] = 10000000
        dist_h_min_r[dist_h_min_r < 0] = 10000000
        dist_r = np.concatenate((dist_h_max_r[:, :, np.newaxis], dist_h_min_r[:, :, np.newaxis],
                                 dist_s_max_r[:, :, np.newaxis], dist_s_min_r[:, :, np.newaxis],
                                 dist_v_max_r[:, :, np.newaxis], dist_v_min_r[:, :, np.newaxis]), axis=2)
        # 根据确定距离最小的来源
        b_pos = np.argmin(dist_b, axis=2)
        r_pos = np.argmin(dist_r, axis=2)

        mask_h_max_b = (b_pos == 0) * index_b
        mask_h_min_b = (b_pos == 1) * index_b
        mask_s_max_b = (b_pos == 2) * index_b
        mask_s_min_b = (b_pos == 3) * index_b
        mask_v_max_b = (b_pos == 4) * index_b
        mask_v_min_b = (b_pos == 5) * index_b

        mask_h_max_r = (r_pos == 0) * index_r
        mask_h_min_r = (r_pos == 1) * index_r
        mask_s_max_r = (r_pos == 2) * index_r
        mask_s_min_r = (r_pos == 3) * index_r
        mask_v_max_r = (r_pos == 4) * index_r

        mask_v_min_r = (r_pos == 5) * index_r

        # 更新值
        h_max_b[mask_h_max_b] = h[mask_h_max_b]
        h_min_b[mask_h_min_b] = h[mask_h_min_b]
        s_max_b[mask_s_max_b] = s[mask_s_max_b]
        s_min_b[mask_s_min_b] = s[mask_s_min_b]
        v_max_b[mask_v_max_b] = v[mask_v_max_b]
        v_min_b[mask_v_min_b] = v[mask_v_min_b]

        h_max_r[mask_h_max_r] = h[mask_h_max_r]
        h_min_r[mask_h_min_r] = h[mask_h_min_r]
        s_max_r[mask_s_max_r] = s[mask_s_max_r]
        s_min_r[mask_s_min_r] = s[mask_s_min_r]
        v_max_r[mask_v_max_r] = v[mask_v_max_r]
        v_min_r[mask_v_min_r] = v[mask_v_min_r]

        # cv2.namedWindow("h_max_b", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("h_min_b", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("s_max_b", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("s_min_b", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("v_max_b", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("v_min_b", cv2.WINDOW_NORMAL)
        #
        # cv2.namedWindow("h_max_r", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("h_min_r", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("s_max_r", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("s_min_r", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("v_max_r", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("v_min_r", cv2.WINDOW_NORMAL)
        #
        # cv2.imshow("h_max_b", h_max_b / 255.0)
        # cv2.imshow("h_min_b", h_min_b / 255.0)
        # cv2.imshow("s_max_b", s_max_b / 255.0)
        # cv2.imshow("s_min_b", s_min_b / 255.0)
        # cv2.imshow("v_max_b", v_max_b / 255.0)
        # cv2.imshow("v_min_b", v_min_b / 255.0)
        #
        # cv2.imshow("h_max_r", h_max_r / 255.0)
        # cv2.imshow("h_min_r", h_min_r / 255.0)
        # cv2.imshow("s_max_r", s_max_r / 255.0)
        # cv2.imshow("s_min_r", s_min_r / 255.0)
        # cv2.imshow("v_max_r", v_max_r / 255.0)
        # cv2.imshow("v_min_r", v_min_r / 255.0)
        #
        # cv2.waitKey()

    # plt.figure(5)
    # plt.plot(np.array(h_max_b_list).transpose())
    plt.show()
    # 保存结果
    root = os.getcwd()
    cv2.imwrite(root + "/models/h_max_b.bmp", h_max_b)  # 保存整个图片
    cv2.imwrite(root + "/models/h_min_b.bmp", h_min_b)
    cv2.imwrite(root + "/models/s_max_b.bmp", s_max_b)
    cv2.imwrite(root + "/models/s_min_b.bmp", s_min_b)
    cv2.imwrite(root + "/models/v_max_b.bmp", v_max_b)
    cv2.imwrite(root + "/models/v_min_b.bmp", v_min_b)
    cv2.imwrite(root + "/models/h_max_r.bmp", h_max_r)
    cv2.imwrite(root + "/models/h_min_r.bmp", h_min_r)
    cv2.imwrite(root + "/models/s_max_r.bmp", s_max_r)
    cv2.imwrite(root + "/models/s_min_r.bmp", s_min_r)
    cv2.imwrite(root + "/models/v_max_r.bmp", v_max_r)
    cv2.imwrite(root + "/models/v_min_r.bmp", v_min_r)

    sz_x = 1520
    sz_y = 3600
    for i in range(5):  # 按照区域保存
        temp = cv2.resize(h_max_b[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/h_max_b_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(h_min_b[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/h_min_b_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(s_max_b[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/s_max_b_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(s_min_b[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/s_min_b_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(v_max_b[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/v_max_b_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(v_min_b[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/v_min_b_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(h_max_r[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/h_max_r_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(h_min_r[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/h_min_r_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(s_max_r[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/s_max_r_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(s_min_r[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/s_min_r_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(v_max_r[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/v_max_r_" + str(4 - i) + ".bmp", temp)

        temp = cv2.resize(v_min_r[:, i * 1550:(i * 1550 + 1520)], (sz_x, sz_y))
        cv2.imwrite(root + "/models/v_min_r_" + str(4 - i) + ".bmp", temp)

    print(np.mean(h_max_b[:]), np.mean(h_min_b[:]), np.mean(s_max_b[:]), np.mean(s_min_b[:]), np.mean(v_max_b[:]),
          np.mean(v_min_b[:]))
    print(np.mean(h_max_r[:]), np.mean(h_min_r[:]), np.mean(s_max_r[:]), np.mean(s_min_r[:]), np.mean(v_max_r[:]),
          np.mean(v_min_r[:]))


def predict():
    h_max_b = cv2.imread("./models/h_max_b.bmp", 0)
    h_min_b = cv2.imread("./models/h_min_b.bmp", 0)
    s_max_b = cv2.imread("./models/s_max_b.bmp", 0)
    s_min_b = cv2.imread("./models/s_min_b.bmp", 0)
    v_max_b = cv2.imread("./models/v_max_b.bmp", 0)
    v_min_b = cv2.imread("./models/v_min_b.bmp", 0)
    h_max_r = cv2.imread("./models/h_max_r.bmp", 0)
    h_min_r = cv2.imread("./models/h_min_r.bmp", 0)
    s_max_r = cv2.imread("./models/s_max_r.bmp", 0)
    s_min_r = cv2.imread("./models/s_min_r.bmp", 0)
    v_max_r = cv2.imread("./models/v_max_r.bmp", 0)
    v_min_r = cv2.imread("./models/v_min_r.bmp", 0)

    # erode 最小值滤波
    # dilate 最大值滤波
    f_size = 10
    h_max_b = cv2.erode(h_max_b, (f_size, f_size))
    s_max_b = cv2.erode(s_max_b, (f_size, f_size))
    v_max_b = cv2.erode(v_max_b, (f_size, f_size))
    h_max_r = cv2.erode(h_max_r, (f_size, f_size))
    s_max_r = cv2.erode(s_max_r, (f_size, f_size))
    v_max_r = cv2.erode(v_max_r, (f_size, f_size))

    h_min_b = cv2.dilate(h_min_b, (f_size, f_size))
    s_min_b = cv2.dilate(s_min_b, (f_size, f_size))
    v_min_b = cv2.dilate(v_min_b, (f_size, f_size))
    h_min_r = cv2.dilate(h_min_r, (f_size, f_size))
    s_min_r = cv2.dilate(s_min_r, (f_size, f_size))
    v_min_r = cv2.dilate(v_min_r, (f_size, f_size))

    # Debug显示
    plt.figure()
    plt.plot(h_max_b[:, 600], 'r')
    plt.plot(h_min_b[:, 600], 'b')
    plt.show()

    # 路径与基本配置
    # image_dir = "/media/cobot/94584AF0584AD0A2/data/normal_images/20190228/left"
    image_dir = "/home/cobot/Desktop/test_color"
    # image_dir = "/home/cobot/Pictures/full/left"
    files = os.listdir(image_dir)
    for name in files:
        print(name)
        # 读图转换为HSV表达
        image = cv2.imread(os.path.join(image_dir, name))
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 判断颜色值是否位于缺陷区域内
        index_b = image_hsv[:, :, 0] < h_max_b
        index_b = index_b * (image_hsv[:, :, 0] > h_min_b)
        index_b = index_b * (image_hsv[:, :, 1] < s_max_b)
        index_b = index_b * (image_hsv[:, :, 1] > s_min_b)
        index_b = index_b * (image_hsv[:, :, 2] < v_max_b+40)
        index_b = index_b * (image_hsv[:, :, 2] > v_min_b - 10)

        index_r = image_hsv[:, :, 0] < h_max_r
        index_r = index_r + (image_hsv[:, :, 0] > h_min_r)
        index_r = index_r * (image_hsv[:, :, 1] < s_max_r)
        index_r = index_r * (image_hsv[:, :, 1] > s_min_r)
        index_r = index_r * (image_hsv[:, :, 2] < v_max_r)
        index_r = index_r * (image_hsv[:, :, 2] > v_min_r)

        # 滤波处理
        # index_b = cv2.blur(index_b.astype(np.float), (5, 5)) > 0.5
        # index_r = cv2.blur(index_r.astype(np.float), (5, 5)) > 0.5
        # index_r = cv2.blur(index_r.astype(np.float), (50, 50) > 0.2

        index = index_b + index_r
        index = index.astype(np.uint8) * 255
        if np.any(index > 0):
            # plt.figure()
            # plt.imshow(index)
            # plt.show()

            # image[index == 255] = 0
            image[index_b] = 0

            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.namedWindow("result_b", cv2.WINDOW_NORMAL)
            cv2.namedWindow("result_r", cv2.WINDOW_NORMAL)

            cv2.imshow("image", image)
            cv2.imshow("result", index)
            cv2.imshow("result_b", index_b.astype(np.uint8) * 255)
            cv2.imshow("result_r", index_r.astype(np.uint8) * 255)

            cv2.waitKey()


def main():
    # find_bounds()
    predict()


if __name__ == "__main__":
    main()
