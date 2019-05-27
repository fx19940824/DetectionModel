#!/usr/bin/python  
# coding:utf-8  

""" 
@author: Yong Li 
@contact: liyong@cobotsys.com
@software: PyCharm 
@file: syn_images.py
@time: 18-11-19 下午1:41
手机螺钉图像合成.从螺钉数据库中读取图像块,根据鼠标位置将螺钉放置在背景图像上.
输入:背景图,系列螺钉图像片,系列螺钉图像片的mask
输出:含螺钉的手机图像,及整张图的mask
操作:需要手动辅助确定螺钉位置,按键N:下一张螺钉图,按键F:确认放置
version:2.0
@time: 18-10-8 下午6:48
添加mask处理功能
添加自动插入螺钉功能

"""

import cv2
import numpy as np
import os
import random
import copy
from ctools.basic_func import get_all_files


# ----------------------支持函数部分-------------------------------------------------------------------------------------
# 支持函数1,图像融合
def merge(image, patch, mask, insert_x, insert_y):
    if image.ndim == 2:
        roi = image[insert_x:(insert_x + patch.shape[0]), insert_y:(insert_y + patch.shape[1])]
    if image.ndim == 3:
        roi = image[insert_x:(insert_x + patch.shape[0]), insert_y:(insert_y + patch.shape[1]), :]
    mask_inv = cv2.bitwise_not(mask)
    # 融合
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(patch, patch, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)

    if image.ndim == 2:
        image[insert_x:(insert_x + patch.shape[0]), insert_y:(insert_y + patch.shape[1])] = dst
    if image.ndim == 3:
        image[insert_x:(insert_x + patch.shape[0]), insert_y:(insert_y + patch.shape[1]), :] = dst
    return image


# ----------------------主程序-------------------------------------------------------------------------------------------
def main(Max_Num=6000, types="screw"):
    break_flag = False  # 跳出循环标志位
    random_flag = True  # 随机旋转的标志位
    batch_flag = True  # 自动批量加入螺钉
    batch_counter = 0
    Max_batch = 20  # 一次性添加的个数

    # Max_Num = 6000  # 合成图像的数量

    step_size = 1  # 放置一个螺钉后 是否调整螺钉

    # types = "screw"  # "screw", "washer", "test" # ToDo 请指定装配类型
    root = "/home/cobot/cellphone_project"

    if types is "screw":
        image_dir = root + '/data/0raw_data/normal/train'  # 指定文件路径
        patch_dir = root + '/data/2patches/screw'
        image_out_dir = "/media/cobot/5C8B2D882D247B561/project_data/screw/simulated/images"
        mask_out_dir = "/media/cobot/5C8B2D882D247B561/project_data/screw/simulated/labels"
    if types is "washer":
        image_dir = root + '/data/0raw_data/normal/train'  # 指定文件路径
        patch_dir = root + '/data/2patches/washer'
        image_out_dir = "/media/cobot/5C8B2D882D247B561/project_data/washer/simulated/images"
        mask_out_dir = "/media/cobot/5C8B2D882D247B561/project_data/washer/simulated/labels"
    if types is "test":
        image_dir = root + '/data/0raw_data/normal/train'  # 指定文件路径
        patch_dir = root + '/data/2patches/screw'
        image_out_dir = "/home/cobot/xxxx/images"
        mask_out_dir = "/home/cobot/xxxx/labels"

    if not os.path.exists(image_out_dir):
        os.makedirs(image_out_dir)
    if not os.path.exists(mask_out_dir):
        os.makedirs(mask_out_dir)

    image_list = get_all_files(image_dir)
    patch_list = get_all_files(patch_dir)
    random.shuffle(image_list)
    random.shuffle(patch_list)

    patch_ind = 0
    image_ind = 0
    output_ind = 0

    while True:  # 不停的读取新图像

        image = cv2.imread(os.path.join(image_dir, image_list[image_ind]))
        if image is None:  # 正常图像读取有问题
            print("Please check the path", os.path.join(image_dir, image_list[image_ind]))
            image_ind = image_ind + 1
            continue
        image = image + 0.0 + np.random.randint(-10, 10)
        np.clip(image, 0, 255, image)
        image = image.astype(np.uint8)

        rect_roi = [0, image.shape[0], 0, image.shape[1]]  # x_min,x_max,y_min,y_max

        image_output = copy.deepcopy(image)  # 用于确定后输出

        mask_output = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        while True:  # 不停的读取新的patch
            if patch_ind >= len(patch_list):  # 检查patch index 是否越界
                patch_ind = 0
            if patch_ind < 0:
                patch_ind = len(patch_list) - 1

            patch = cv2.imread(os.path.join(patch_dir, patch_list[patch_ind]))

            if patch is None:  # 螺钉文件不存在,或者有问题
                print("Please check the path", os.path.join(patch_dir, patch_list[patch_ind]))
                patch_ind = patch_ind + 1
                break

            _, mask = cv2.threshold(cv2.imread(os.path.join(patch_dir, patch_list[patch_ind]), 0), 254, 255,
                                    cv2.THRESH_BINARY_INV)  # 确定mask

            # 随机旋转4个角度,做数据增强 Todo 更多的数据增强的手段
            if random_flag:
                k = np.random.randint(0, 4)
                patch = np.rot90(patch, k)
                mask = np.rot90(mask, k)
                if np.random.rand() < 0.5:
                    patch = np.flipud(patch)
                    mask = np.flipud(mask)
                patch = patch + 0.0 + np.random.randint(-10, 10)
                np.clip(patch, 0, 255, patch)
                patch = patch.astype(np.uint8)

            # 自动批量添加螺钉
            if batch_flag and batch_counter < Max_batch:
                assemble_x = np.random.randint(rect_roi[0], rect_roi[1] - patch.shape[0])
                assemble_y = np.random.randint(rect_roi[2], rect_roi[3] - patch.shape[1])
                assemble_flag = True

            assemble_x = np.clip(assemble_x, rect_roi[0], rect_roi[1] - patch.shape[0])  # 确定插入点
            assemble_y = np.clip(assemble_y, rect_roi[2], rect_roi[3] - patch.shape[1])

            # image_temp = merge(copy.copy(image_output), patch, mask, assemble_x, assemble_y)  # 图像融合

            # 放置螺钉
            if assemble_flag:
                assemble_flag = False
                patch_ind = patch_ind + step_size
                image_output = merge(image_output, patch, mask, assemble_x, assemble_y)  # 图像融合
                mask_output = merge(mask_output, mask, mask, assemble_x, assemble_y)  # 图像融合
                if batch_flag:
                    batch_counter = batch_counter + 1

            # 保存图像
            if batch_counter == Max_batch:
                batch_counter = 0
                new_name = image_list[image_ind].split(os.sep)[-1][:-4] + '_' + str(100000 + output_ind) + ".png"
                cv2.imwrite(os.path.join(image_out_dir, new_name), image_output)
                cv2.imwrite(os.path.join(mask_out_dir, new_name), mask_output)

                image_ind = image_ind + 1
                output_ind = output_ind + 1
                print(types, str(output_ind) + "...")
                break  # 退出添加新螺钉

            # 退出程序
            if output_ind == Max_Num:
                print("退出!")
                break_flag = True
                break

        # 检测指标,图像指标是否越界
        if image_ind >= len(image_list):
            image_ind = 0
        if image_ind < 0:
            image_ind = len(image_list)

        if break_flag:  # 结束所有
            break


if __name__ == "__main__":
    main(Max_Num=8000, types="washer")
