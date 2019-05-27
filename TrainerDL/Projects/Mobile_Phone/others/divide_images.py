#!/usr/bin/python
# coding:utf-8

"""
@author: Yong Li
@contact: liyong@cobotsys.com
@software: PyCharm
@file: syn_images.py
@time: 19-03-8 下午1:41
图像按照类别转移；
操作:直接下一张图,按键N:转移到文件夹，按键T；

"""

import cv2
import numpy as np
import os
import copy
import shutil

image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/fixed_phones/left"
out_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/20190228/left"


# ----------------------主程序-------------------------------------------------------------------------------------------
def main():
    image_list = os.listdir(image_dir)
    cv2.namedWindow("Main window", cv2.WINDOW_GUI_NORMAL)
    image_ind = 0
    process_flag = True
    while True and image_ind < len(image_list):
        if process_flag:
            image = cv2.imread(os.path.join(image_dir, image_list[image_ind]))
            if image is None:  # 正常图像读取有问题
                print("Please check the path", os.path.join(image_dir, image_list[image_ind]))
                image_ind = image_ind + 1
                continue
            process_flag = False

        cv2.imshow("Main window", image)
        key = cv2.waitKey(1) & 0xFF
        # if key < 255:
        #     print(key)

        if key == ord('n'):  # 直接下1张图像
            image_ind = image_ind + 1
            process_flag = True
            print("直接下一张图像！")

        if key == ord('t'):  # 转移后，显示下一张图像
            shutil.move(os.path.join(image_dir, image_list[image_ind]), os.path.join(out_dir, image_list[image_ind]))
            image_ind = image_ind + 1
            process_flag = True
            print("转移后，下一张图像！")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
