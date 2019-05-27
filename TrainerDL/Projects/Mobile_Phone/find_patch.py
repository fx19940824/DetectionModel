#!/usr/bin/python
# coding:utf-8

import os
import cv2
import random
import numpy as np
from ctools.basic_func import get_all_files

roi_length = [48, 148]


for j, left_right in enumerate(["left", "right"]):
    # 模板文件地址和样本文件夹
    mask_dir = "/home/cobot/Desktop/1_" + left_right
    image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/20190312/" + left_right
    output_dir = "/home/cobot/Desktop/temp"

    mask_files = get_all_files(mask_dir)
    rois = []
    for mask_file in mask_files:
        mask = cv2.imread(os.path.join(mask_dir, mask_file), 0)
        bool_img = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        _, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(bool_img[1], 8, cv2.CV_16U,
                                                                                    cv2.CCL_WU)
        batch = len(centroids)
        for centers, stat in zip(centroids, stats):
            rois = rois + [centers[0::1] - roi_length]

    files = get_all_files(image_dir)
    random.shuffle(files)
    files = files[0:40]  # 样本数量设置

    for k, file_name in enumerate(files):
        print(str(k) + ":" + file_name)
        image = cv2.imread(os.path.join(image_dir, file_name))
        for i, roi in enumerate(rois[1:]):
            roi = roi.astype(np.int)
            # print([roi[1], (roi[1] + roi_length[1] * 2), roi[0], (roi[0] + roi_length[0] * 2)])
            patch = image[roi[1]:(roi[1] + roi_length[1] * 2), roi[0]:(roi[0] + roi_length[0] * 2)]
            full_path = os.path.join(output_dir, str(j) + "_" + str(i) + os.sep + file_name.split(os.sep)[-1])
            if not os.path.exists(os.path.dirname(full_path)):
                os.mkdir(os.path.dirname(full_path))
            cv2.imwrite(full_path, patch)
