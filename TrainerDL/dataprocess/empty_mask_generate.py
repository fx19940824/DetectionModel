# coding=utf-8
import glob
import numpy as np
import cv2
from tqdm import tqdm

#创建文件夹
if __name__ == '__main__':
	# 读取路径
	file_list = glob.glob('/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/Data/mask/*.png')
	imglist = []

	for filename in tqdm(file_list):
		image = cv2.imread(filename, cv2.IMREAD_COLOR)
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.imwrite(filename.replace('.png', '_mask.png'), mask)
