#!/usr/bin/python
# coding:utf-8

"""
@author: Yong Li
@contact: liyong@cobotsys.com
@software: PyCharm
@file: final1.py
@time: 18-9-8 上午11:33
更新时间:2018年11月23日
富士康手机螺钉垫片检测模块
"""
import numpy as np
import cv2
import copy
import os
from ctools.basic_func import image_correct, get_all_files
from skimage.measure import label, regionprops

import time
import sys

root = "/home/cobot/cellphone_project"


# 利用相似性对比，判断少螺钉
class ScrewMiss(object):
    def __init__(self):
        # 处理ROIs,记录中间点
        self.rois = []
        self.roi_length = 42
        self.get_rois()
        self.batch = len(self.rois)

        # 得到正负模板（合成图）
        self.good_patches = [[0] * 8 for row in range(36)]
        self.bad_patches = [[0] * 3 for row in range(36)]
        self.get_patches()

        # todo

    def processing(self, img):
        image = np.zeros(img.shape[0:2], dtype=np.uint8)
        mask_output = np.zeros(img.shape[0:2], dtype=np.uint8)
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, image)

        warp_mode = cv2.MOTION_TRANSLATION
        number_of_iterations = 100
        termination_eps = 1e-8
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        for k, roi in enumerate(self.rois):
            patch = image[roi[0]:roi[0] + self.roi_length, roi[1]:roi[1] + self.roi_length]
            # good for
            g_cc = 0.0
            for i in range(8):
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                try:
                    (cc, M) = cv2.findTransformECC(patch, self.good_patches[k][i], warp_matrix, warp_mode, criteria)
                except:
                    cc = 0
                if cc > g_cc:
                    g_cc = cc

            # bad for
            b_cc = 0.0
            for i in range(3):
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                # cv2.imwrite("patch.png", patch)
                # cv2.imwrite("xxxx.png", self.bad_patches[k][i])
                try:
                    (cc, M) = cv2.findTransformECC(patch, self.bad_patches[k][i], warp_matrix, warp_mode, criteria)
                except:
                    cc = 0
                if cc > b_cc:
                    b_cc = cc

            # 判断正常缺少螺钉的条件
            threshold = 0.666
            epsilon = 0.05
            print
            k, g_cc, b_cc
            if g_cc + epsilon < b_cc or g_cc < threshold or b_cc > 0.96:  # or g_cc < b_cc - epsilon:  # 缺少螺钉的图像
                mask_output[roi[0]:roi[0] + self.roi_length, roi[1]:roi[1] + self.roi_length] = 1
            pass
        return mask_output

    def get_rois(self):
        self.rois = []
        mask = cv2.imread("/home/cobot/cellphone_project/data/3templates/missing_screw3.png", 0)
        bool_img = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(bool_img[1])
        self.batch = len(centroids)
        for centers, stat in zip(centroids, stats):
            if stat[4] < 10000:
                self.rois = self.rois + [centers[::-1].astype(np.int) - self.roi_length / 2]

    def get_patches(self):
        for k in range(len(self.rois)):
            good_path = os.path.join(root + "/data/3templates/miss_screw/good", str(k))
            bad_path = os.path.join(root + "/data/3templates/miss_screw/bad", str(k))

            i = 0
            for good_short in os.listdir(good_path):
                # print k, i
                self.good_patches[k][i] = cv2.imread(os.path.join(good_path, good_short), 0)  # 读取正常的图像
                i = i + 1
            i = 0
            for bad_short in os.listdir(bad_path):
                # print k, i
                self.bad_patches[k][i] = cv2.imread(os.path.join(bad_path, bad_short), 0)  # 读取正常的图像
                i = i + 1


def test():
    sm = ScrewMiss()

    image = cv2.imread("/home/cobot/Desktop/1217pingbi/zhengchagn/2018.12.18_10:57:10.595_.png")
    template = cv2.imread("/home/cobot/cellphone_project/data/3templates/2.png")

    image, M = image_correct(template, image)
    mask = sm.processing(image)
    cv2.imwrite("mask.png", (mask + 0.0) * 255)
    cv2.imwrite("image.png", image)


def test_all():
    sm = ScrewMiss()

    # image_dir = "/media/cobot/5C8B2D882D247B561/project_data/normal/case7"
    image_dir = "/home/cobot/Desktop/xxxxxx"
    files = get_all_files(image_dir)
    template = cv2.imread("/home/cobot/cellphone_project/data/3templates/2.png")

    for k, path in enumerate(files):
        print(k)
        print(path)

        image = cv2.imread(os.path.join(image_dir, path))

        image, M = image_correct(template, image)
        mask = sm.processing(image)
        if np.max(mask) > 0:
            print(path)
            cv2.imwrite("/home/cobot/Desktop/tempx/" + path, image)
            cv2.imwrite("/home/cobot/Desktop/tempx/" + path[:-4] + "_r.png", mask * 255)
            # raw_input("xxx:")


# test_all()
test()
