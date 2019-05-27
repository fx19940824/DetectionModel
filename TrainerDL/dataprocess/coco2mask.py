from pycocotools.coco import COCO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO 根据 coco格式的标注数据生成每个图片对应的mask
path = '/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/Data'

annFile = os.path.join(path, 'annotations/instances_Data_train.json')
coco = COCO(annFile)

Ids = coco.getAnnIds()
catIds = coco.getCatIds(catNms=['1'])
imgIds = coco.getImgIds(catIds=1)
for id in tqdm(imgIds):
    img = coco.getImgIds(imgIds=id)
    img = coco.loadImgs(img)[0]
    img_path = os.path.join(path, 'train', img['file_name'])
    I = cv2.imread(img_path)

    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    annIds = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annIds)
    mask = np.zeros(shape=(img['height'],img['width']), dtype=np.uint8)
    for id, ann in enumerate(anns):
        mask += coco.annToMask(ann)*255
        # mask = cv2.medianBlur(mask, 11)

    # cv2.imshow(',', mask)
    # cv2.imshow('.', I)
    # I[mask==0]=0
    cv2.imwrite(img_path.replace('.png', '_mask.png'), mask)


annFile = os.path.join(path, 'annotations/instances_Data_val.json')
coco = COCO(annFile)


catIds = coco.getCatIds(catNms=['1'])
imgIds = coco.getImgIds(catIds=1)
for id in tqdm(imgIds):
    img = coco.getImgIds(imgIds=id)
    img = coco.loadImgs(img)[0]
    img_path = os.path.join(path, 'val', img['file_name'])
    I = cv2.imread(img_path)

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    mask = np.zeros(shape=(img['height'], img['width']), dtype=np.uint8)
    for id, ann in enumerate(anns):
        mask += coco.annToMask(ann)*255
    # cv2.imshow(',', mask)
    # cv2.imshow('.', I)
    # cv2.waitKey(0)
    cv2.imwrite(img_path.replace('.png', '_mask.png'), mask)