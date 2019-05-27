#!/usr/bin/python  
# coding:utf-8  

""" 
@author: Yong Li 
@contact: liyong@cobotsys.com
@software: PyCharm 
@file: extract_masks.py 
@time: 18-11-28 下午8:01
"""

import numpy as np
import cv2
import os

import matplotlib.pyplot as plt

img_dir = "/home/cobot/Desktop/_3miss_screw/label0315"


names = os.listdir(img_dir)
names.sort()

mask = []
cv2.namedWindow("xx", cv2.WINDOW_GUI_NORMAL)

for k, name in enumerate(names):
    print(str(k) + ":" + name)
    img = cv2.imread(os.path.join(img_dir, name))
    # plt.imshow(img)
    # plt.pause(1.0)
    if img is None:
        continue

    if np.max(img) == 255:  # 有mask,读取mask
        mask = (img > 254)
        cv2.imwrite(os.path.join(img_dir, name), mask)
    else:  # 没有mask,则用前面的mask
        img[mask] = 255
        cv2.imshow("xx", img)
        cv2.waitKey(100)
        cv2.imwrite(os.path.join(img_dir, name), img)

    pass


def main():
    pass


if __name__ == "__main__":
    main()
