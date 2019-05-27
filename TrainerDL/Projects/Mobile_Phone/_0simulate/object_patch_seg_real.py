# #!/usr/bin/python
# # coding:utf-8
#
# """
# @author: Yong Li
# @contact: liyong@cobotsys.com
# @software: PyCharm
# @file: object_patch_seg_real.py
# @time: 18-11-19 下午4:42
# 功能描述:从真实数据中扣取螺钉Patches
# """
# import cv2
# import numpy as np
# import os
# from skimage.measure import label, regionprops
# from ctools.basic_func import get_all_files
#
# # ---------------主程序部分-------------------------------------------------------------------------------------------
# root = ""
# image_dir = "/media/cobot/5C8B2D882D247B56/pic_beifen/1207-data/chunluoding"  # 原始图目录
# mask_dir = "/home/cobot/Desktop/请标注4/chunluoding"  # 标签目录
# output_dir = "/home/cobot/Desktop/yyy"  # 输出位置
#
# files = get_all_files(image_dir)
#
# new_ind = 0
# for short_name in files:
#     # short_name = "2018.10.30_17:18:24.679_7f8aac9030f0.png"
#     # 读图和mask
#     image = cv2.imread(os.path.join(image_dir, short_name))
#     image[image > 253] = 253
#     image[image < 1] = 1
#     mask = cv2.imread(os.path.join(mask_dir, short_name), 0)
#     if image is None or mask is None:
#         print short_name + ": image does not exist or reading error!"
#         continue
#     if np.any(mask % 255):
#         print short_name + ": label image should be binary!!!"
#
#     # mask分析和处理
#     label_image = label(mask)
#     props = regionprops(label_image)
#
#     # 根据mask的blob的裁剪图像
#     for k in range(len(props)):
#         cps = np.array(props[k].centroid).astype(np.int)
#         if cps[0] in range(54, image.shape[0] - 54) and cps[1] in range(54, image.shape[1] - 54):
#             image_patch = image[(cps[0] - 53):cps[0] + 53, cps[1] - 53:cps[1] + 53, :]
#             mask_patch = label_image[(cps[0] - 53):cps[0] + 53, cps[1] - 53:cps[1] + 53] == k + 1
#
#             mask_patch = mask_patch[:, :, np.newaxis] * image_patch
#             mask_patch[mask_patch == 0] = 255
#             print mask_patch.shape
#
#             new_ind = new_ind + 1
#             res = cv2.imwrite(output_dir + os.sep + short_name.split(os.sep)[-1][:-4] + "_p_" + str(new_ind) + ".png",
#                               mask_patch)
#             # res = cv2.imwrite(output_dir + os.sep + "xxx" + "_p_" + str(new_ind) + ".png",
#             #                   mask_patch)
#             # cv2.imshow("xx", mask_patch)
#             # cv2.waitKey(10)
#
# print(files)
