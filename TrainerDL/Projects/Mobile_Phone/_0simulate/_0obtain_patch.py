#!/usr/bin/python
# coding:utf-8

"""
@author: Yong Li
@contact: liyong@cobotsys.com
@software: PyCharm
@file: object_patch_seg_real.py
@time: 19-02-28 下午4:42
功能描述:从真实数据中扣取螺钉Patches
"""
import cv2
import numpy as np
import os
# from skimage.measure import label, regionprops
from ctools.basic_func import get_all_files

# ---------------主程序部分-------------------------------------------------------------------------------------------
root = ""
image_dir = "/media/cobot/94584AF0584AD0A2/data/_2screw_patches/第五次标注/多垫片/分/壳"  # 原始图目录
mask_dir = "/media/cobot/94584AF0584AD0A2/data/_2screw_patches/第五次标注/多垫片标注/分/壳"  # 标签目录
output_dir = "/home/cobot/Desktop/sss"  # 输出位置

files = get_all_files(image_dir)

new_ind = 0
for short_name in files:
    print(short_name)
    # Step.1 读图和mask
    image = cv2.imread(os.path.join(image_dir, short_name))
    mask = cv2.imread(os.path.join(mask_dir, short_name), 0)
    # cv2.imshow("mask", mask)
    # cv2.waitKey()

    if image is None or mask is None:
        print(short_name + ": image does not exist or reading error!")
        continue
    mask = cv2.inRange(mask, 254, 255)
    if np.any(mask % 255):
        print(short_name + ": label image should be binary!!!")

    # Step.2 mask分析和处理
    _, label_image, props, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

    # 根据mask的blob的裁剪图像
    for k in range(len(centroids)):
        cps = centroids[k, :]
        area = props[k, 4]
        if 500 < area < 400000 and 110 < cps[0] < image.shape[1] - 110 and 110 < cps[1] < image.shape[0] - 110:
            cps = np.round(cps[::-1]).astype(np.int)
            image_patch = image[(cps[0] - 100):cps[0] + 100, cps[1] - 100:cps[1] + 100, :]
            mask_patch = 255 * (label_image[(cps[0] - 100):cps[0] + 100, cps[1] - 100:cps[1] + 100] == k)
            mask_patch = mask_patch.astype(np.uint8)

            # 合成4通道patch图像
            image_patch = np.concatenate((image_patch, mask_patch[:, :, np.newaxis]), axis=2)
            res = cv2.imwrite(output_dir + os.sep + "hhh" + "_p_" + str(1e6 + new_ind) + ".png", image_patch)
            new_ind = new_ind + 1
            if new_ind % 100 == 0:
                print("合成数据" + str(new_ind) + "张")

# print(files)
