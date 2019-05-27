# coding=utf-8
import glob
import cv2
import sys
import os

#创建文件夹
def mkdir(path):
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

def maincreate(name,imgname,dir_output):
	image = cv2.imread(name+".png",cv2.IMREAD_COLOR)
	array = image
	cv2.imwrite(dir_output + '/' +imgname+'.png', array)
	h_flip = cv2.flip(array, 1)
	cv2.imwrite(dir_output + '/' +imgname+'_h.png', h_flip)
	v_flip = cv2.flip(array, 0)
	cv2.imwrite(dir_output + '/' +imgname+'_v.png', v_flip)
	# Flipped Horizontally & Vertically 水平垂直翻转
	hv_flip = cv2.flip(array, -1)
	cv2.imwrite(dir_output + '/' +imgname+'_hv.png', hv_flip)

if __name__=='__main__':
	# 读取路径
	dir_origin = '/home/cobot/TEST/no-bomo/'
	dir_output = '/home/cobot/mixdata/overturn'
	mkdir(dir_output)
	extension = 'png'

	file_list = glob.glob(dir_origin + '/*.' + extension)
	imglist = []
	for index, filename in enumerate(file_list):
		str_index = str(index)
		theimgname = filename.split('/')[-1]
		theimgname = theimgname.split('.png')[0]
		filepath = filename.split('.png')[0]
		imglist.append((filepath, theimgname))

	thenum = 0
	thetotal = len(imglist)
	for key1,key2 in imglist:
		thenum += 1
		maincreate(key1,key2,dir_output)
		sys.stdout.write('{0}/{1}\r'.format(thenum,thetotal))
		sys.stdout.flush()


	print("total:"+str(thetotal))