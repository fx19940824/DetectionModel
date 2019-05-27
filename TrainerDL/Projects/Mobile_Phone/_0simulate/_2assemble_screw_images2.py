#!/usr/bin/python  
# coding:utf-8  

""" 
@author: Yong Li 
@contact: liyong@cobotsys.com
@software: PyCharm 
@file: syn_images.py
@time: 18-10-8 下午1:41
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
import copy

# 全局变量
assemble_x = 0  # 插入点坐标,同insert_x,insert_y
assemble_y = 0
assemble_flag = False  # 确定是否插入


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


# ----------------------回调函数-----------------------------------------------------------------------------------------
def cvMouseCallback(event, x, y, flags, param):
    global assemble_x, assemble_y, assemble_flag

    # print type(param[0]), param[1]
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下
        assemble_x = y
        assemble_y = x
        assemble_flag = True
        pass
    elif event == cv2.EVENT_MOUSEMOVE:  # 鼠标移动
        assemble_x = y
        assemble_y = x
        pass
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        pass
    elif event == cv2.EVENT_MBUTTONDOWN:  # 鼠标中间按下
        pass

    # 调整下,是mask大致位于中心
    # assemble_x = assemble_x - 50
    # assemble_y = assemble_y - 50


# ----------------------主程序-------------------------------------------------------------------------------------------
def main():
    global assemble_x, assemble_y, assemble_flag  # 全局变量

    break_flag = False  # 跳出循环标志位
    random_flag = True  # 随机旋转的标志位
    batch_flag = False  # 自动批量加入螺钉,暂定为100颗
    batch_counter = 0
    Max_batch = 40  # 一次性添加的个数

    rect_roi = [390, 1180, 340, 1930]  # x_min,x_max,y_min,y_max

    step_size = 1  # 放置一个螺钉后 是否调整螺钉

    image_dir = '/media/yong/data/cellphone_inspention/cellphone/python2/images/normal_images'  # 指定文件路径
    patch_dir = '/media/yong/data/cellphone_inspention/cellphone/python2/images/screw_patches'
    image_out_dir = '/media/yong/data/cellphone_inspention/cellphone/python2/images/output/images'
    mask_out_dir = '/media/yong/data/cellphone_inspention/cellphone/python2/images/output/labels'

    image_list = os.listdir(image_dir)
    patch_list = os.listdir(patch_dir)

    patch_ind = 0
    image_ind = 0
    output_ind = 0

    cv2.namedWindow("Main window", cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback("Main window", cvMouseCallback)

    while True:
        image = cv2.imread(os.path.join(image_dir, image_list[image_ind]))
        if image is None:  # 正常图像读取有问题
            print("Please check the path", os.path.join(image_dir, image_list[image_ind]))
            image_ind = image_ind + 1
            continue

        image_temp = copy.deepcopy(image)  # 用于随时显示
        image_output = copy.deepcopy(image)  # 用于确定后输出

        mask_output = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        while True:
            patch = cv2.imread(os.path.join(patch_dir, patch_list[patch_ind]))
            if patch is None:  # 螺钉文件不存在,或者有问题
                patch_ind = patch_ind + 1
                print("Please check the path", os.path.join(patch_dir, patch_list[patch_ind]))
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

            # 自动批量添加螺钉
            if batch_flag and batch_counter < Max_batch:
                assemble_x = np.random.randint(rect_roi[0], rect_roi[1] - patch.shape[0])
                assemble_y = np.random.randint(rect_roi[2], rect_roi[3] - patch.shape[1])
                assemble_flag = True

            assemble_x = np.clip(assemble_x, rect_roi[0], rect_roi[1] - patch.shape[0])  # 确定插入点
            assemble_y = np.clip(assemble_y, rect_roi[2], rect_roi[3] - patch.shape[1])

            image_temp = merge(copy.copy(image_output), patch, mask, assemble_x, assemble_y)  # 图像融合

            cv2.imshow("Main window", image_temp)
            key = cv2.waitKey(1) & 0xFF
            # if key < 255:
            #     print(key)

            if key == ord('s'):  # 下1颗螺钉
                patch_ind = patch_ind + 1
                print("下箭头")
                pass
            if key == ord('w'):  # 上1颗螺钉
                patch_ind = patch_ind - 1
                print("上箭头")
                pass
            if key == ord('a'):  # 下10颗螺钉
                print("左箭头")
                patch_ind = patch_ind + 10
                pass
            if key == ord('d'):  # 上10颗螺钉
                print("右箭头")
                patch_ind = patch_ind - 10
                pass
            if key == ord('r'):  # 随机旋转和反转模式
                print("随机旋转翻转模式!")
                random_flag = not random_flag
                pass
            if key == ord('m'):
                print("调整步进模式")
                if step_size == 0:
                    step_size = 1
                elif step_size == 1:
                    step_size = 0
                pass
            if key == ord('b'):
                print("自动添加模式")
                batch_flag = True
                batch_counter = 0
                pass
            if key == 13:  # 完成该图的放置,开启下一张图
                print("回车确认")
                # 保存结果
                new_name = image_list[image_ind][:-4] + '_' + str(100000 + output_ind) + image_list[image_ind][-4:]
                cv2.imwrite(os.path.join(image_out_dir, new_name), image_output)
                cv2.imwrite(os.path.join(mask_out_dir, new_name), mask_output)

                image_ind = image_ind + 1
                output_ind = output_ind + 1

                break  # 退出添加新螺钉
                pass
            if key == 27:  # ESC按键退出程序
                print("退出!")
                break_flag = True
                break
            pass

            # 放置螺钉
            if assemble_flag:
                assemble_flag = False
                patch_ind = patch_ind + step_size
                image_output = merge(image_output, patch, mask, assemble_x, assemble_y)  # 图像融合
                mask_output = merge(mask_output, mask, mask, assemble_x, assemble_y)  # 图像融合
                if batch_flag:
                    batch_counter = batch_counter + 1

            if patch_ind >= len(patch_list):
                patch_ind = 0
            if patch_ind < 0:
                patch_ind = len(patch_list) - 1

        # 检测指标,图像指标是否越界
        if image_ind >= len(image_list):
            image_ind = 0
        if image_ind < 0:
            image_ind = len(image_list)

        if break_flag:  # 结束所有
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# image = cv2.imread("/media/yong/data/cellphone_inspention/cellphone/python2/images/_2018-07-05_10-14-36.jpg")
# patch = cv2.imread("/media/yong/data/cellphone_inspention/cellphone/python2/images/xx.jpeg")
#
# patch_mask = cv2.imread("/media/yong/data/cellphone_inspention/cellphone/python2/images/xx.jpeg", 0)
#
# # 指定roi,mask
# insert_x = np.random.randint(0, image.shape[0] - patch.shape[0])
# insert_y = np.random.randint(0, image.shape[1] - patch.shape[1])
# _, mask = cv2.threshold(patch_mask, 100, 255, cv2.THRESH_BINARY_INV)
#
# # 图像融合
# image = merge(image, patch, mask, insert_x, insert_y)
#
# # mask融合
# # mask_inv = cv2.bitwise_not(mask)
# #
# # # 融合
# # # Now black-out the area of logo in ROI
# # img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# # # Take only region of logo from logo image.
# # img2_fg = cv2.bitwise_and(patch, patch, mask=mask)
# # # Put logo in ROI and modify the main image
# # dst = cv2.add(img1_bg, img2_fg)
# #
# # image[insert_x:(insert_x + patch.shape[0]), insert_y:(insert_y + patch.shape[1]), :] = dst
#
# # cv2.imshow("mask", mask.astype(np.uint8))
#
# cv2.imshow("xxx", image)
# cv2.waitKey()
# # img1[0:rows, 0:cols] = dst
# #
# # # I want to put logo on top-left corner, So I create a ROI
# # rows, cols, channels = img2.shape
# # roi = img1[0:rows, 0:cols]
# # # Now create a mask of logo and create its inverse mask also
# # img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# # ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
# # mask_inv = cv2.bitwise_not(mask)
# # # Now black-out the area of logo in ROI
# # img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# # # Take only region of logo from logo image.
# # img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
# # # Put logo in ROI and modify the main image
# # dst = cv2.add(img1_bg, img2_fg)
# # img1[0:rows, 0:cols] = dst
# # cv2.imshow('res', img1)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # patch嵌入到一张空白图中
# blank_img = np.zeros(image.shape, np.uint8)
# blank_mask = np.zeros(image.shape[0:2], np.uint8)
# insert_x = np.random.randint(0, image.shape[0] - patch.shape[0])
# insert_y = np.random.randint(0, image.shape[1] - patch.shape[1])
#
# blank_img[insert_x:(insert_x + patch.shape[0]), insert_y:(insert_y + patch.shape[1]), :] = patch
# blank_mask[insert_x:(insert_x + patch.shape[0]), insert_y:(insert_y + patch.shape[1])] = patch_mask
#
# # get first masked value (foreground)
# fg = cv2.bitwise_or(blank_img, blank_img, mask=blank_mask)
#
# # get second masked value (background) mask must be inverted
# # mask = cv2.bitwise_not(mask)
# # background = np.full(img.shape, 255, dtype=np.uint8)
# bk = cv2.bitwise_or(image, image, mask=cv2.bitwise_not(blank_mask))
#
# # combine foreground+background
# final = cv2.bitwise_or(fg, bk)
#
# cv2.imshow("mask", blank_mask.astype(np.uint8) * 255)
# cv2.imshow("blank_img", blank_img)
# cv2.imshow("bk", bk)
# cv2.imshow("final", final)
# cv2.waitKey()
#
# # def merge(img, patch, patch_mask, center):
# #     return img, mask
# #
# #
#
# #
# #
# # if __name__ == "__main__":
# #     main()
