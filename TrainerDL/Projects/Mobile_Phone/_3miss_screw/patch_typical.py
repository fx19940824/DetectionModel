#!/usr/bin/python
# coding:utf-8

import os
import cv2
import numpy as np
import shutil
from ctools.basic_func import get_all_files

input_dir_o = "/home/cobot/Desktop/bad"
output_dir = "/home/cobot/Desktop/bad2"

for input_dirs in os.listdir(input_dir_o):
    input_dir = os.path.join(input_dir_o, input_dirs)
    files = os.listdir(input_dir)

    warp_mode = cv2.MOTION_TRANSLATION
    number_of_iterations = 100
    termination_eps = 1e-8
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    results = np.zeros((len(files), len(files)))
    for i in range(len(files)):
        img1 = cv2.imread(os.path.join(input_dir, files[i]))
        img1_gray = np.zeros(img1.shape[0:2], dtype=np.uint8)
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY, img1_gray)
        for j in range(i, len(files)):
            img2 = cv2.imread(os.path.join(input_dir, files[j]))
            img2_gray = np.zeros(img2.shape[0:2], dtype=np.uint8)
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY, img2_gray)

            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                (cc, M) = cv2.findTransformECC(img1_gray, img2_gray, warp_matrix, warp_mode, criteria)
            except:
                cc = 0
            results[i, j] = cc
            results[j, i] = cc

    numbers = 20
    ind = [0] * numbers
    # find the first one
    ind[0] = np.argmax(np.sum(results, axis=0))

    # find the next one
    temp = results[ind[0], :]
    for i in range(numbers - 1):
        results[:, ind[i]] = 1
        ind[i + 1] = np.argmin(temp)
        temp = temp + results[ind[i + 1], :]

    for i in range(5):
        print(i, ind[i], files[ind[i]])
        src = os.path.join(input_dir, files[ind[i]])
        dst = os.path.join(output_dir, input_dirs + os.sep + files[ind[i]])
        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))
        shutil.copy(src, dst)

        # ind[2] = np.argmin(results[ind[0], :] + results[ind[1], :])
