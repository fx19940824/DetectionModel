import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm


coco = COCO('/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/CN-DATA/BAG/train/Data/annotations/instances_Data_train.json')

Ids = coco.getAnnIds()

width_list = []
height_list = []

radio_list = []
num_list = []
for annIds in Ids:
    anns = coco.loadAnns(annIds)
    bbox = anns[0]['bbox']
    width_list.append(bbox[2]/1.8)
    height_list.append(bbox[3]/1.8)
    radio_list.append(bbox[2]/bbox[3])
    num_list.append(annIds)

# plt.subplot(121)
plt.xlim(xmax = 500, xmin = 0)
plt.ylim(ymax = 500, ymin = 0)
plt.title('bordingbox')
plt.xlabel("width")
plt.ylabel('height')
plt.plot(width_list, height_list, '.')
plt.show()
# plt.subplot(122)
plt.xlim(xmax = 12000, xmin = 0)
plt.ylim(ymax = 5, ymin = 0)
plt.title('bordingbox')
plt.xlabel("num")
plt.ylabel('radio')
plt.plot(num_list, radio_list, '.')


plt.show()