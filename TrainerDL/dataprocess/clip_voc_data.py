import xml.etree.cElementTree as et
import os
import re
import cv2

def getInfo(path):
    root = et.parse(path)
    object_node = root.findall('object')
    size_node = root.findall('size')
    cords = []
    size = []
    labelName = []

    for node in object_node:
        object_node_child = node.getchildren()
        for object_node_child_element in object_node_child:
            if object_node_child_element.tag == 'name':
                labelitem = object_node_child_element.text
                labelName.append(labelitem)
            if object_node_child_element.tag == 'bndbox':
                xmin = object_node_child_element[0].text
                ymin = object_node_child_element[1].text
                xmax = object_node_child_element[2].text
                ymax = object_node_child_element[3].text
                cords.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    size_node_child = size_node[0].getchildren()
    for size_node_child_element in size_node_child:
        if size_node_child_element.tag == 'width':
            width = size_node_child_element.text
        if size_node_child_element.tag == 'height':
            height = size_node_child_element.text
    size.append({'width': width, 'height': height})

    return labelName, cords, size

path = '/home/eleven/FLL-VAL'

for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith('.xml'):
            xmlPath = root + '/' + name
            labelName, cords, size = getInfo(xmlPath)

            img_name = os.path.splitext(name)[0]
            imgPath = root + '/' + img_name +".jpg"
            img = cv2.imread(imgPath)

            for label, cord in zip(labelName, cords):
                img_clip_path = root + '/' + img_name + "_" + cord['xmin'] \
                                + "_" +cord['ymin'] + '_' + cord['xmax'] + '_' \
                                + cord['ymax'] + '_yolo.jpg'
                print(img_clip_path)
                img_clip = img[int(cord['ymin']): int(cord['ymax']), int(cord['xmin']): int(cord['xmax'])]
                cv2.imwrite(img_clip_path, img_clip)

