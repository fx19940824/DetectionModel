# coding=utf-8
import glob
import os
import numpy as np
import json
import cv2
from tqdm import tqdm

if __name__ == '__main__':
	# 读取路径
	file_list = glob.glob('/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/Data/mask/*.png')
	imglist = []
	for filename in tqdm(file_list):
		jsonname = filename.replace('.png', '.json')
		if (not os.path.exists(jsonname)):
			continue

		image = cv2.imread(filename, cv2.IMREAD_COLOR)
		f = open(jsonname, encoding='utf-8')
		myjson = json.load(f)
		if myjson['shapes'] == []:
			continue
		a = myjson['shapes'][0]['points']  # a是标记的点
		b = np.array(a, dtype=np.int32)
		roi_t = []
		for i in range(len(a)):
			roi_t.append(b[i])
		roi_t = np.asarray(roi_t)
		roi_t = np.expand_dims(roi_t, axis=0)
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.polylines(mask, roi_t, 1, 255)
		cv2.fillPoly(mask, roi_t, 255)

		cv2.imwrite(filename.replace('.png', '_mask.png'), mask)
