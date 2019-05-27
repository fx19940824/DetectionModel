# coding=utf-8
import glob
import os
import numpy as np
import json
import cv2
import copy
import sys
import base64

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

def main(name,imgname,dir_output,scale=0.5):
    image = cv2.imread(name + ".png", cv2.IMREAD_COLOR)
    f = open(name + ".json", encoding='utf-8')
    myjson = json.load(f)
    if myjson['shapes'] == []:
        return

    for shape in myjson['shapes']:
        points=shape['points']
        for i in range(len(points)):
            points[i][0]*=scale
            points[i][1]*=scale

    rows, cols, channels = image.shape
    array = cv2.resize(image, (int(scale * cols), int(scale * rows)))
    cv2.imwrite(dir_output + '/' + imgname + '_zoom.png', array)
    myjson["imagePath"] = dir_output + '/' + imgname + '_zoom.png'
    # myjson.pop("imageData")

    with open(dir_output + '/' + imgname + '_zoom.png', 'rb') as f:
        imageData = f.read()
        myjson["imageData"] = base64.b64encode(imageData).decode('utf-8')

    with open(dir_output + '/' + imgname + '_zoom.json', 'w', encoding='utf-8') as json_file:
        json.dump(myjson, json_file, ensure_ascii=False)

if __name__=='__main__':
    dir_source = '/home/cobot/bomo_all'
    dir_output = '/home/cobot/mixdata/split_zoom'
    scale=0.25
    mkdir(dir_output)
    extension = 'png'
    imglist = []
    for index, filename in enumerate(glob.glob(dir_source + '/*.' + extension)):
        str_index = str(index)
        theimgname = filename.split('/')[-1]
        theimgname = theimgname.split('.png')[0]
        filepath = filename.split('.png')[0]
        imglist.append((filepath, theimgname))

    thenum = 0
    thetotal = len(imglist)
    for key1, key2 in imglist:
        thenum += 1
        main(key1, key2, dir_output, scale)
        sys.stdout.write('{0}/{1}\r'.format(thenum, thetotal))
        sys.stdout.flush()

    print("total:" + str(thetotal))