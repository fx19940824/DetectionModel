import cv2
import json
import base64
import glob
from tqdm import tqdm


# TODO mask图片转labelmer标注文件
annotation = {
    "version": "3.9.0",
    "flags": {},
    "shapes": [
        {
            "label": "0",
            "line_color": "null",
            "fill_color": "null",
            "points": [],
            "shape_type": "polygon"
        }
    ],
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
    "imageHeight": 660,
    "imageWidth": 473,
}
masks = glob.glob("/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/Data/mask/*mask.png")

for mask in tqdm(masks):
    print(mask)
    img = mask.replace("_mask.png", ".png")
    f = open(img, 'rb')
    image_data = base64.b64encode(f.read()).decode('utf-8')
    img_mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    cnt, _ = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for points in cnt:
        area = cv2.contourArea(points)
        if area > 1000:
            points = points.squeeze().tolist()
            points = points[::20]
            annotation["shapes"][0]["points"] = points
            annotation["imagePath"] = img
            annotation["imageData"] = image_data
        jsonstring = json.dumps(annotation, indent=2)
        jsonstring = jsonstring.replace("\"null\"", "null")
        open(img.replace(".png",".json"), "w").write(jsonstring)