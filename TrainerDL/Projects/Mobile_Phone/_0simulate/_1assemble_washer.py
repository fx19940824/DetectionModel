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
@time:19-02-28 上午10:40
更新为双相机合成图像功能;
label需要填充过渡区域；
label设置规则：0-背景；128-过渡区域；255-螺钉

"""

# TODO label需要填充过渡区域；

import sys

sys.path.append("/home/cobot/mobile_project/python")
import cv2
import numpy as np
import os
import random
import copy
from ctools.basic_func import get_all_files

AREA = -1
# 设置全局变量,路径问题
left_image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/left"
patch_dir = "/media/cobot/94584AF0584AD0A2/data/_4washer_patches"

if AREA < 0:
    left_out_dir = "/media/cobot/8e505336-96a8-4c09-bc62-9ca728a68af3/mobile_washer_train_data/left"
    mask_out_dir = "/media/cobot/8e505336-96a8-4c09-bc62-9ca728a68af3/mobile_washer_train_data/label"
else:
    left_out_dir = "/home/cobot/mobile_data/screw_train/left" + str(AREA)
    mask_out_dir = "/home/cobot/mobile_data/screw_train/label" + str(AREA)


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
def asemble(Max_Num=6000, types="screw"):
    break_flag = False  # 跳出循环标志位
    random_flag = True  # 随机旋转的标志位
    batch_flag = True  # 自动批量加入螺钉
    batch_counter = 0
    # Max_batch = 20  # 一次性添加的个数
    if AREA > -1:
        Max_batch = 4  # 一次性添加的个数
    else:
        Max_batch = 20  # 一次性添加的个数

    # Max_Num = 6000  # 合成图像的数量

    step_size = 1  # 放置一个螺钉后 是否调整螺钉

    # if types is "screw":
    #     image_dir = root + '/data/0raw_data/normal/train'  # 指定文件路径
    #     patch_dir = root + '/data/2patches/screw'
    #     image_out_dir = "/media/cobot/5C8B2D882D247B561/project_data/screw/simulated/images"
    #     mask_out_dir = "/media/cobot/5C8B2D882D247B561/project_data/screw/simulated/labels"
    # if types is "washer":
    #     image_dir = root + '/data/0raw_data/normal/train'  # 指定文件路径
    #     patch_dir = root + '/data/2patches/washer'
    #     image_out_dir = "/media/cobot/5C8B2D882D247B561/project_data/washer/simulated/images"
    #     mask_out_dir = "/media/cobot/5C8B2D882D247B561/project_data/washer/simulated/labels"
    # if types is "test":
    #     image_dir = root + '/data/0raw_data/normal/train'  # 指定文件路径
    #     patch_dir = root + '/data/2patches/screw'
    #     image_out_dir = "/home/cobot/xxxx/images"
    #     mask_out_dir = "/home/cobot/xxxx/labels"

    if not os.path.exists(left_out_dir):  # 检测路径输入输入
        os.makedirs(left_out_dir)
    if not os.path.exists(mask_out_dir):
        os.makedirs(mask_out_dir)

    image_list = get_all_files(left_image_dir)
    patch_list = get_all_files(patch_dir)
    random.shuffle(image_list)
    random.shuffle(patch_list)

    patch_ind = 0
    image_ind = 0
    output_ind = 0

    while True:  # 不停的读取新图像
        # 检测指标,图像指标是否越界
        if image_ind >= len(image_list):
            image_ind = 0
        if image_ind < 0:
            image_ind = len(image_list)

        left_image = cv2.imread(os.path.join(left_image_dir, image_list[image_ind]), -1)
        if left_image is None:  # 正常图像读取有问题
            print("Please check the path", os.path.join(left_image_dir, image_list[image_ind]))
            image_ind = image_ind + 1
            continue
        if AREA > -1:
            i = 4 - AREA
            left_image = left_image[:, i * 1550:(i * 1550 + 1520), :]

        rect_roi = [0, left_image.shape[0], 0, left_image.shape[1]]  # x_min,x_max,y_min,y_max

        left_output = copy.deepcopy(left_image)  # 用于确定后输出

        mask_output = np.zeros((left_output.shape[0], left_output.shape[1]), dtype=np.uint8)

        while True:  # 不停的读取新的patch
            if patch_ind >= len(patch_list) - 2:  # 检查patch index 是否越界
                patch_ind = 0
            if patch_ind < 0:
                patch_ind = len(patch_list) - 1

            patch1 = cv2.imread(os.path.join(patch_dir, patch_list[patch_ind]), -1)

            if patch1 is None:  # 螺钉文件不存在,或者有问题
                print("Please check the path", os.path.join(patch_dir, patch_list[patch_ind]))
                patch_ind = patch_ind + 1
                break

            mask1 = patch1[:, :, 3]

            # 随机旋转4个角度,做数据增强 Todo 更多的数据增强的手段
            if random_flag:
                k = np.random.randint(0, 4)  # 随机旋转
                patch1 = np.rot90(patch1, k)
                mask1 = np.rot90(mask1, k)

            if np.random.rand() < 0.5:
                patch1 = np.flipud(patch1)
                mask1 = np.flipud(mask1)

            patch1 = patch1[:, :, 0:3] + 0.0 + np.random.randint(-10, 10)
            np.clip(patch1, 0, 255, patch1)
            patch1 = patch1.astype(np.uint8)

            # 自动批量添加螺钉
            if batch_flag and batch_counter < Max_batch:
                assemble_x = np.random.randint(rect_roi[0], rect_roi[1] - patch1.shape[0])
                assemble_y = np.random.randint(rect_roi[2], rect_roi[3] - patch1.shape[1])
                assemble_flag = True

            assemble_x = np.clip(assemble_x, rect_roi[0], rect_roi[1] - patch1.shape[0])  # 确定插入点
            assemble_y = np.clip(assemble_y, rect_roi[2], rect_roi[3] - patch1.shape[1])

            # image_temp = merge(copy.copy(image_output), patch, mask, assemble_x, assemble_y)  # 图像融合

            # 放置螺钉
            if assemble_flag:
                assemble_flag = False
                patch_ind = patch_ind + step_size

                left_output = merge(left_output, patch1, mask1, assemble_x, assemble_y)  # 图像融合
                mask_output = merge(mask_output, mask1, mask1, assemble_x, assemble_y)  # 标签融合

                if batch_flag:
                    batch_counter = batch_counter + 1

            # 保存图像
            if batch_counter == Max_batch:
                # Mask生成过渡区域
                # 中间模糊地带的数据,通过形态学open操作得到label_open,并与模板匹配变形
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 150))  # 形态学处理下，避免负样本中包含螺钉
                label_open = cv2.morphologyEx(np.array(mask_output == 255).astype(np.uint8), cv2.MORPH_DILATE,
                                              kernel)

                label_result = 128 * np.ones(mask_output.shape, dtype=np.uint8)  # 默认值为128,模糊地带不做处理
                label_result[mask_output == 255] = 255  # 有螺钉值为1
                label_result[label_open != 1] = 0  # 没有螺钉,背景值为0

                # 写入图像
                batch_counter = 0
                new_name = image_list[image_ind].split(os.sep)[-1][:-4] + '_' + str(1e8 + output_ind) + ".bmp"
                cv2.imwrite(os.path.join(left_out_dir, new_name), left_output)
                cv2.imwrite(os.path.join(mask_out_dir, new_name), label_result)

                image_ind = image_ind + 1
                output_ind = output_ind + 1
                print(types, str(output_ind) + "...")
                break  # 退出添加新螺钉

            # 退出程序
            if output_ind == Max_Num:
                print("退出!")
                break_flag = True
                break

        if break_flag:  # 结束所有
            break


if __name__ == "__main__":
    asemble(Max_Num=3, types="washer")
