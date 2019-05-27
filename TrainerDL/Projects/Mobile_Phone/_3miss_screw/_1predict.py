import os
import cv2
import random
import numpy as np
from ctools.basic_func import get_all_files
import glob


# 检测mask中blob的中心点
def get_center_points(mask):
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    _, _, _, center_points = cv2.connectedComponentsWithStats(mask)
    center_points = center_points[1:, :]
    center_points = np.array(np.round(center_points))  #
    center_points = center_points[:, ::-1]
    return center_points


# step 1.0 读取正负模板
good_dir = "/home/cobot/Desktop/good2"
bad_dir = "/home/cobot/Desktop/bad2"

weight = cv2.imread("weight.bmp", 0)
weight = weight.astype(np.float) / 255

rois_number = [15, 0, 1, 3, 11]
good_image_list = []
bad_image_list = []
for left_right in [0, 1]:
    good_image_list.append([])
    bad_image_list.append([])
    for area in range(5):
        good_image_list[left_right].append([])
        bad_image_list[left_right].append([])
        for roi in range(rois_number[area]):
            good_image_list[left_right][area].append([])
            bad_image_list[left_right][area].append([])
            for image_number in range(5):
                # good
                dir_path = os.path.join(good_dir, str(left_right) + str(area) + "_" + str(roi))
                names = os.listdir(dir_path)[0:5]
                for name in names:
                    image = cv2.imread(os.path.join(dir_path, name), 0)
                    good_image_list[left_right][area][roi].append(image)
                # bad
                dir_path = os.path.join(bad_dir, str(left_right) + str(area) + "_" + str(roi))
                names = os.listdir(dir_path)[0:5]
                for name in names:
                    image = cv2.imread(os.path.join(dir_path, name), 0)
                    bad_image_list[left_right][area][roi].append(image)

# step 2.0 模板对比
left_mask_dir = "/home/cobot/caid2.0/data/deploy/mobile_phone_screw/templates/miss_screw/masks_left"
right_mask_dir = "/home/cobot/caid2.0/data/deploy/mobile_phone_screw/templates/miss_screw/masks_right"
left_roi_list = []
right_roi_list = []
for area in range(5):
    left_mask = cv2.imread(os.path.join(left_mask_dir, "mask" + str(area) + ".bmp"), 0)
    right_mask = cv2.imread(os.path.join(right_mask_dir, "mask" + str(area) + ".bmp"), 0)
    _, left_bool = cv2.threshold(left_mask, 254, 255, cv2.THRESH_BINARY)
    _, right_bool = cv2.threshold(right_mask, 254, 255, cv2.THRESH_BINARY)

    # n_component, _, stats_left, centr_left = cv2.connectedComponentsWithStats(left_bool)
    # _, _, stats_right, centr_right = cv2.connectedComponentsWithStats(right_bool)
    n_component, _, stats_left, centr_left = cv2.connectedComponentsWithStatsWithAlgorithm(left_bool, 8, cv2.CV_16U,
                                                                                           cv2.CCL_WU)
    _, _, stats_right, centr_right = cv2.connectedComponentsWithStatsWithAlgorithm(right_bool, 8, cv2.CV_16U,
                                                                                   cv2.CCL_WU)
    # xxx = np.concatenate([centr_left, centr_right], axis=1)
    index_left = stats_left[:, 4] < 20000
    index_right = stats_right[:, 4] < 20000
    left_roi = centr_left[index_left]
    right_roi = centr_right[index_right]
    # left_roi = np.sort(left_roi, 0)
    index_left = np.argsort(left_roi[:, 1] + left_roi[:, 0] / 10)
    index_right = np.argsort(right_roi[:, 1] + right_roi[:, 0] / 10)
    left_roi = left_roi[index_left]
    right_roi = right_roi[index_right]

    left_roi_list.append(np.round(left_roi).astype(np.int))
    right_roi_list.append(np.round(right_roi).astype(np.int))


def cal_simi(img, template):
    # cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    # cv2.namedWindow("temp", cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow("image", img)
    # cv2.imshow("temp", template)
    # cv2.waitKey()
    # print(img.shape, template.shape)
    image = img.copy() + 0.0
    temp = template.copy() + 0.0
    image = image * weight
    temp = temp * weight
    image = image - np.mean(image[:])
    image = image / np.sqrt(np.sum(image[:] * image[:]) + 0.0000000001)
    temp = temp - np.mean(temp[:])
    temp = temp / np.sqrt(np.sum(temp[:] * temp[:]) + 0.0000000001)
    cc = np.sum(image[:] * temp[:])
    return cc


# Step 3.0读图预测
# left_image_dir = "/home/cobot/Desktop/_3miss_screw/left"
# right_image_dir = "/home/cobot/Desktop/_3miss_screw/right"
# label_dir = "/home/cobot/Desktop/_3miss_screw/label"

left_image_dir = "/home/cobot/Desktop/temp/left"
right_image_dir = "/home/cobot/Desktop/temp/right"
label_dir = ""

# left_image_dir = "/home/cobot/Desktop/test_miss_screw/left"
# right_image_dir = "/home/cobot/Desktop/test_miss_screw/right"

# left_image_dir = "/media/cobot/5C8B2D882D247B56/mobile_special_data/miss_screw/left"
# right_image_dir = "/media/cobot/5C8B2D882D247B56/mobile_special_data/miss_screw/right"
# left_image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/left/20190228"
# right_image_dir = "/media/cobot/94584AF0584AD0A2/data/_0normal_images/right/20190228"
# label_dir = ""


for name in os.listdir(left_image_dir)[0::1]:
    print(name)
    image = cv2.imread(os.path.join(left_image_dir, name))
    left_image = cv2.imread(os.path.join(left_image_dir, name), 0)
    right_image = cv2.imread(os.path.join(right_image_dir, name), 0)
    label_image = cv2.imread(os.path.join(label_dir, name), 0)
    result = np.zeros_like(left_image)
    for area in range(5):
        left_img = left_image[:, (4 - area) * 1550: ((4 - area) * 1550 + 1520)]
        right_img = right_image[:, (4 - area) * 1550: ((4 - area) * 1550 + 1520)]
        result_roi = result[:, (4 - area) * 1550: ((4 - area) * 1550 + 1520)]
        for k in range(len(left_roi_list[area])):
            left_roi = left_roi_list[area][k]
            right_roi = right_roi_list[area][k]
            left_img_roi = left_img[left_roi[1] - 64:left_roi[1] + 64, left_roi[0] - 64:left_roi[0] + 64]
            right_img_roi = right_img[right_roi[1] - 64:right_roi[1] + 64, right_roi[0] - 64:right_roi[0] + 64]
            gcc1 = 0.0
            bcc1 = 0.0
            gcc2 = 0.0
            bcc2 = 0.0
            for i in range(5):
                left_good_template = good_image_list[0][area][k][i]
                right_good_template = good_image_list[1][area][k][i]
                left_bad_template = bad_image_list[0][area][k][i]
                right_bad_template = bad_image_list[1][area][k][i]
                cc = cal_simi(left_img_roi, left_good_template)
                gcc1 = np.maximum(gcc1, cc)
                cc = cal_simi(right_img_roi, right_good_template)
                gcc2 = np.maximum(gcc2, cc)
                # print("gcc", i, gcc1, gcc2)

                cc = cal_simi(left_img_roi, left_bad_template)
                bcc1 = np.maximum(bcc1, cc)
                cc = cal_simi(right_img_roi, right_bad_template)
                bcc2 = np.maximum(bcc2, cc)
                # print("bcc", i, gcc1, gcc2)

                pass
            epsilon = 0.05
            gcc = np.maximum(gcc1, gcc2)
            bcc = np.maximum(bcc1, bcc2)

            # if area == 0 and k == 1:
            #     print(gcc1, gcc2, bcc1, bcc2)
            #     cv2.imshow("goodtemp", good_image_list[1][area][k][0])
            #     cv2.imshow("badtemp", bad_image_list[1][area][k][0])
            #     cv2.imshow("img", right_img_roi)
            # cv2.waitKey()
            if bcc - gcc > 0.05 and bcc > 0.3:
                print(area, k, bcc - gcc, gcc1, gcc2, bcc1, bcc2)
                #     cv2.imwrite("images/" + name + str(area) + str(k) + "l.bmp", left_img_roi)
                #     cv2.imwrite("images/" + name + str(area) + str(k) + "r.bmp", right_img_roi)
                #
                result_roi[right_roi[1] - 64:right_roi[1] + 64, right_roi[0] - 64:right_roi[0] + 64] = 255
                #     # result[:, (4 - area) * 1550: ((4 - area) * 1550 + 1520)] = result_roi
                #     # cv2.imshow("roi", result_roi)
                #     # if area == 3 and k == 0:
                # cv2.imshow("goodtemp", good_image_list[1][area][k][0])
                # cv2.imshow("badtemp", bad_image_list[1][area][k][0])
                # cv2.imshow("img", right_img_roi)
                # cv2.waitKey()
    if np.sum(result[:]) >= 0:
        label_centers = get_center_points(label_image)
        result_centers = get_center_points(result)
        for center in label_centers:
            cv2.circle(image, (center[1].astype(np.int), center[0].astype(np.int)), 100, (255, 0, 0), thickness=5)
        for center in result_centers:
            cv2.circle(image, (center[1].astype(np.int), center[0].astype(np.int)), 80, (0, 0, 255), thickness=5)

        cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow("result", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("image", image)
        cv2.imshow("result", result)
        cv2.waitKey()
