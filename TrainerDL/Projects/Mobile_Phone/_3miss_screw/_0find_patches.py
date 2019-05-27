#!/usr/bin/python
# coding:utf-8

import os
import cv2
import random
import numpy as np
from ctools.basic_func import get_all_files

for j, left_right in enumerate(["left", "right"]):

    for area in range(5):
        # area = 0
        # image_dir = "/media/cobot/5C8B2D882D247B561/project_data/screw_miss/1217"
        # image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/" + left_right + "/20190309"
        # image_dir = "/home/cobot/Desktop/_3miss_screw/" + left_right + ""
        image_dir = "/home/cobot/Desktop/temp/" + left_right + ""
        output_dir = "/home/cobot/Desktop/temp"

        # get rois
        roi_length = 128
        rois = []
        mask = cv2.imread(
            "/home/cobot/caid2.0/data/deploy/mobile_phone_screw/templates/miss_screw/masks_" + left_right + "/mask" + str(
                area) + ".bmp", 0)
        bool_img = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        # _, labels, stats, centroids = cv2.connectedComponentsWithStats(bool_img[1])
        _, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(bool_img[1], 8, cv2.CV_16U,
                                                                                    cv2.CCL_WU)
        batch = len(centroids)
        for centers, stat in zip(centroids, stats):
            if stat[4] < 500000:
                rois = rois + [centers[::-1] - roi_length / 2]
                # rois = rois + [centers[::-1]]
        # #
        # if True:
        #     for k, centers in enumerate(rois):
        #         cv2.circle(mask, (centers[1].astype(np.int), centers[0].astype(np.int)), 100, (255, 255, 255), 10)
        #         # cv2.putText(mask, "xx" + str(k), (centers[1].astype(np.int), centers[0].astype(np.int)),
        #         #             cv2.FONT_HERSHEY_SIMPLEX, 20.0, (255, 255, 255),
        #         #             lineType=cv2.LINE_AA)
        #
        #     cv2.imwrite("xxx.png", mask)

        files = get_all_files(image_dir)
        random.shuffle(files)
        files = files[0:50]
        for k, file_name in enumerate(files):
            print(str(k) + ":" + file_name)
            image = cv2.imread(os.path.join(image_dir, file_name))[:, (4 - area) * 1550:((4 - area) * 1550 + 1520), :]
            # for k, centers in enumerate(rois):
            #     cv2.circle(image, (centers[1].astype(np.int), centers[0].astype(np.int)), 100, (255, 255, 255), 10)
            #     # cv2.putText(mask, "xx" + str(k), (centers[1].astype(np.int), centers[0].astype(np.int)),
            #     #             cv2.FONT_HERSHEY_SIMPLEX, 20.0, (255, 255, 255),
            #     #             lineType=cv2.LINE_AA)
            # cv2.imshow("xx", image)
            # cv2.waitKey();

            # image, M = image_correct(template, image)
            for k, roi in enumerate(rois):
                roi = roi.astype(np.int)
                patch = image[roi[0]:roi[0] + roi_length, roi[1]:roi[1] + roi_length]
                full_path = os.path.join(output_dir,
                                         str(j) + str(area) + "_" + str(k) + os.sep + file_name.split(os.sep)[-1])
                if not os.path.exists(os.path.dirname(full_path)):
                    os.mkdir(os.path.dirname(full_path))
                cv2.imwrite(full_path, patch)
