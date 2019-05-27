import cv2
import matplotlib.pyplot as plt
import random
import json
import time
import glob
import os
import sys
import base64
import copy
from tqdm import tqdm


def mkdir(path):
	import os
	path=path.strip()
	path=path.rstrip("\\")
	isExists=os.path.exists(path)
	if not isExists:
		os.makedirs(path) 
		print(path+' 创建成功')
		return True
	else:
		print(path+' 目录已存在')
		return False

def mainadd(sourceimg,splitimg,splitjson,newname,dir_output):
	img1 = cv2.imread(sourceimg)
	img2 = cv2.imread(splitimg)

	f = open(splitjson, encoding='utf-8')
	myjson = json.load(f)
	a = myjson['shapes'][0]['points']  # a是标记的点
	# img1为要添加的图片，img2为薄膜

	rows, cols, channels = img2.shape
	middle_img2_rows = int(rows / 2)
	middle_img2_cols = int(cols / 2)
	maxrows, maxcols, img1channels = img1.shape

	randomrows = random.randint(0, maxrows - rows - 1)
	randomcols = random.randint(0, maxcols - cols - 1)

	roi = img1[randomrows : randomrows + rows, randomcols : randomcols + cols]  # 选取随机区域

	# 创建掩膜
	img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)

	# 保留除logo外的背景
	img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
	dst = cv2.add(img1_bg, img2)  # 进行融合

	img1[randomrows : randomrows + rows,
		  randomcols : randomcols + cols] = dst  # 融合后放在原图上

	# img1 = 0.2 * copyimg + 0.8 * img1
	cv2.imwrite(dir_output + '/' + newname + '.png', img1)
	myjson["imagePath"] = dir_output + '/' + newname + '.png'
	
	for i in range(len(a)):
		a[i][0] = a[i][0] + randomcols
		a[i][1] = a[i][1] + randomrows
	myjson['shapes'][0]['points'] = a
	with open(dir_output + '/' + newname + '.png', 'rb') as f:
		imageData = f.read()
		myjson["imageData"] = base64.b64encode(imageData).decode('utf-8')

	with open(dir_output + '/' + newname + '.json', 'w', encoding='utf-8') as json_file:
		json.dump(myjson, json_file, ensure_ascii=False)

if __name__ == '__main__':

	#读取路径
	dir_split = '/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/split/'
	dir_source = '/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/source/'
	dir_output = '/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/out/'
	# mkdir(dir_output)
	n_total = 100

	file_list = glob.glob(dir_split + '*.png')
	splitlist = []
	for index, filename in enumerate(file_list):
		str_index = str(index)
		filepath = filename.split('.png')[0]
		splitlist.append(filepath)

	file_list = glob.glob(dir_source + '*.png')
	sourcelist = []
	for index, filename in enumerate(file_list):
		str_index = str(index)
		filepath = filename.split('.png')[0]
		sourcelist.append(filepath)

	splitnum = len(splitlist)
	sourcenum = len(sourcelist)
	thenum = 0
	for i in tqdm(range(n_total)):
		now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
		splitrandom = random.randint(0,splitnum-1)
		sourcerandom = random.randint(0,sourcenum-1)
		splitimg = str(splitlist[splitrandom]) + '.png'
		sourceimg = str(sourcelist[sourcerandom]) + '.png'
		splitjson = str(splitlist[splitrandom]) + '.json'
		newname = now + str(i)
		mainadd(sourceimg,splitimg,splitjson,newname,dir_output)
		thenum += 1
		sys.stdout.write('{0}/{1}\r'.format(thenum,n_total))
		sys.stdout.flush()

	print("total:"+ str(n_total))
