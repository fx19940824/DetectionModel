import cv2
import numpy as np
import shutil
import os
from ctools.basic_func import get_all_files

image_dir = "/home/cobot/mobile_project/python/_4mylar/models"

names = get_all_files(image_dir)
for k, name in enumerate(names):
    print(k, name)
    image = cv2.imread(os.path.join(image_dir, name))
    cv2.imwrite(os.path.join(image_dir, name)[:-4] + ".jpg", image)

    # mask_dir = "/home/cobot/Pictures/full/miss_screw"
    # left_image_dir = "/home/cobot/Pictures/full/left"
    # right_image_dir = "/home/cobot/Pictures/full/right"
    #
    # left_out_dir = "/home/cobot/Desktop/test_miss_screw/left"
    # right_out_dir = "/home/cobot/Desktop/test_miss_screw/right"
    #
    # names = os.listdir(mask_dir)
    # for name in names:
    #     shutil.copy(os.path.join(left_image_dir, name), os.path.join(left_out_dir, name))
    #     shutil.copy(os.path.join(right_image_dir, name), os.path.join(right_out_dir, name))

    #
    # #
    # # dir_name = "/home/cobot/Desktop/请标注2/多螺钉/right"
    # #
    # # names = os.listdir(dir_name)
    # #
    # # for k, name in enumerate(names):
    # #     print(k, name)
    # #     full_name = os.path.join(dir_name, name)
    # #     image = cv2.imread(full_name)
    # #     image[image > 253] = 253
    # #     cv2.imwrite(full_name, image)
    #
    # image = cv2.imread("/home/cobot/caid2.0/Debug/bin/rois0_14.bmp", 0) + 0.0
    # templ = cv2.imread(
    #     "/home/cobot/caid2.0/data/deploy/mobile_phone_screw/templates/miss_screw/bad/0_14/2019.03.05_11.00.19.001_.bmp",
    #     0) + 0.0
    #
    # # image = image - np.mean(image[:])
    # xx = image / np.sqrt(np.sum(image[:] * image[:]))
    # yy = np.sum(xx[:]*xx[:])
    # templ = templ - np.mean(templ[:])
    #
    # # templ = image
    #
    # cc = np.sum(image[:] * templ[:]) / np.sqrt(np.sum(image[:] * image[:]) * (np.sum(templ[:] * templ[:])))
    # print(cc)
