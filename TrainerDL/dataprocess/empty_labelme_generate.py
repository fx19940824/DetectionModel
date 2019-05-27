import cv2
import json
import base64
import glob

# TODO 根据图像生成无目标标注的labelme标注文件
annotation = {
    "version": "3.9.0",
    "flags": {},
    "shapes": [],
    "lineColor": [
        0,
        255,
        0,
        128
    ],
    "fillColor": [
        255,
        0,
        0,
        128
    ],
    "imagePath": "",
    "imageHeight": 0,
    "imageWidth": 0,
}
imgs = glob.glob("/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/Data/val/*.png")

for imgpath in imgs:
    img = cv2.imread(imgpath)
    f = open(imgpath, 'rb')
    image_data = base64.b64encode(f.read()).decode('utf-8')
    annotation["imagePath"] = imgpath
    annotation["imageData"] = image_data
    annotation["imageHeight"] = img.shape[0]
    annotation["imageWidth"] = img.shape[1]
    jsonstring = json.dumps(annotation, indent=2)
    jsonstring = jsonstring.replace("\"null\"", "null")
    open(imgpath.replace(".png",".json"), "w").write(jsonstring)