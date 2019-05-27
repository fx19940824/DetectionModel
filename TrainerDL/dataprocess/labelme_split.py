# coding=utf-8
import glob
import os
import numpy as np
import json
import cv2
import copy
import sys
from tqdm import tqdm

#创建文件夹
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

def mainsplit(name,imgname,dir_output):
	kernel = np.ones((3, 3), np.uint8)
	image = cv2.imread(name+".png",cv2.IMREAD_COLOR)
	f = open(name+".json", encoding='utf-8')
	myjson = json.load(f)
	if myjson['shapes'] == []:
		return 0 
	a = myjson['shapes'][0]['points']#a是标记的点
	a_1 = copy.deepcopy(a)
	list_col = []
	list_row = []
	for key in a:
		list_col.append(key[0])
		list_row.append(key[1])
	maxcol = int(max(list_col))
	mincol = int(min(list_col))
	maxrow = int(max(list_row))
	minrow = int(min(list_row))
	lena_1 = len(a_1)
	for i in range(lena_1):
		a_1[i][0] = a_1[i][0] - mincol
		a_1[i][1] = a_1[i][1] - minrow
	newshape = []
	newshape.append(myjson['shapes'][0])
	myjson['shapes'] = newshape
	myjson['shapes'][0]['points'] = a_1
	
	b = np.array(a,dtype = np.int32)
	roi_t = []
	for i in range(len(a)):
		roi_t.append(b[i])
	roi_t = np.asarray(roi_t)
	roi_t = np.expand_dims(roi_t, axis=0)
	im = np.zeros(image.shape[:2], dtype = "uint8")
	cv2.polylines(im, roi_t, 1, 255)
	cv2.fillPoly(im, roi_t, 255)
	mask = im
	masked = cv2.bitwise_and(image, image, mask=mask)
	array = np.zeros((masked.shape[0], masked.shape[1], 4), np.uint8)
	array[:, :, 0:3] = masked
	array[:, :, 3] = 0
	array[:,:,3][np.where(array[:,:,0]>=1)]=255
	array[:,:,3][np.where(array[:,:,1]>=1)]=255
	array[:,:,3][np.where(array[:,:,2]>=1)]=255
	array = array[minrow:maxrow,mincol:maxcol,:]

	# array_erode = cv2.erode(array, kernel)
	cv2.imwrite(dir_output + '/' +imgname+'_split.png',array)
	myjson["imagePath"] = dir_output + '/' +imgname+'_split.png'
	#myjson.pop("imageData")
	with open(dir_output + '/' +imgname+'_split.json','w',encoding='utf-8') as json_file:
		json.dump(myjson,json_file,ensure_ascii=False)
	thex,they = array.shape[:2]
	# print(thex,they)
	a_2 = copy.deepcopy(a_1)
	# Flipped Horizontally 水平翻转
	h_flip = cv2.flip(array, 1)
	h_flip[:,:,3][np.where(h_flip[:,:,0]>=1)]=255
	h_flip[:,:,3][np.where(h_flip[:,:,1]>=1)]=255
	h_flip[:,:,3][np.where(h_flip[:,:,2]>=1)]=255
	# h_flip=cv2.erode(h_flip, kernel)
	cv2.imwrite(dir_output + '/' +imgname+'_split-h.png', h_flip)
	for i in range(lena_1):
		a_2[i][0] = they - a_2[i][0]
	myjson['shapes'][0]['points'] = a_2
	myjson["imagePath"] = dir_output + '/' +imgname+'_split-h.png'
	with open(dir_output + '/' +imgname+'_split-h.json','w',encoding='utf-8') as json_file:
		json.dump(myjson,json_file,ensure_ascii=False)

	a_2 = copy.deepcopy(a_1)
	# Flipped Vertically 垂直翻转
	v_flip = cv2.flip(array, 0)
	v_flip[:,:,3][np.where(v_flip[:,:,0]>2)]=255
	v_flip[:,:,3][np.where(v_flip[:,:,1]>2)]=255
	v_flip[:,:,3][np.where(v_flip[:,:,2]>2)]=255
	v_flip = cv2.erode(v_flip, kernel)
	cv2.imwrite(dir_output + '/' +imgname+'_split-v.png', v_flip)
	for i in range(lena_1):
		a_2[i][1] = thex - a_2[i][1]
	myjson['shapes'][0]['points'] = a_2
	myjson["imagePath"] = dir_output + '/' +imgname+'_split-v.png'
	with open(dir_output + '/' +imgname+'_split-v.json','w',encoding='utf-8') as json_file:
		json.dump(myjson,json_file,ensure_ascii=False)


	# Flipped Horizontally & Vertically 水平垂直翻转
	hv_flip = cv2.flip(array, -1)
	hv_flip[:,:,3][np.where(hv_flip[:,:,0]>2)]=255
	hv_flip[:,:,3][np.where(hv_flip[:,:,1]>2)]=255
	hv_flip[:,:,3][np.where(hv_flip[:,:,2]>2)]=255
	hv_flip = cv2.erode(hv_flip, kernel)
	cv2.imwrite(dir_output + '/' +imgname+'_split-hv.png', hv_flip)
	for i in range(lena_1):
		a_2[i][0] = they - a_2[i][0]
	myjson['shapes'][0]['points'] = a_2
	myjson["imagePath"] = dir_output + '/' +imgname+'_split-hv.png'
	with open(dir_output + '/' +imgname+'_split-hv.json','w',encoding='utf-8') as json_file:
		json.dump(myjson,json_file,ensure_ascii=False)

if __name__ == '__main__':
	# 读取路径
	dir_output = '/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/split/'
	mkdir(dir_output)
	file_list = glob.glob('/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/bag/*.png')
	imglist = []
	for filename in tqdm(file_list):
		jsonname = filename.replace('.png', '.json')

		if (not os.path.exists(jsonname)):
			continue

		theimgname = filename.split('/')[-1]
		theimgname = theimgname.split('.png')[0]
		filepath = filename.split('.png')[0]
		imglist.append((filepath, theimgname))

	thenum = 0
	thetotal = len(imglist)
	for key1,key2 in tqdm(imglist):
		thenum += 1
		mainsplit(key1,key2,dir_output)
		sys.stdout.write('{0}/{1}\r'.format(thenum,thetotal))
		sys.stdout.flush()


	print("total:"+str(thetotal))
