import json
import os

from glob import glob, iglob
import re
import base64
from shutil import rmtree
from PIL import Image
import argparse
from numpy import random
from tqdm import tqdm

# exit()


def empty_dir(path):
    """ empty specified dir """
    if os.path.exists(path):
        rmtree(path)
    os.mkdir(path)


def ensure_dir(path):
    """ empty specified dir """
    if not os.path.exists(path):
        os.mkdir(path)


def get_bbox(coords):
    """ get bounding box in format [tlx, tly, w, h] """
    min_x = None
    min_y = None
    max_x = None
    max_y = None

    for [x, y] in coords:
        min_x = x if not min_x else min(x, min_x)
        min_y = y if not min_y else min(y, min_y)
        max_x = x if not max_x else max(x, max_x)
        max_y = y if not max_y else max(y, max_y)

    return [min_x, min_y, max_x - min_x, max_y - min_y]

def generate_dataset(INPUT_DIR, OUTPUT_DIR, suffix='Data', ratio=0, isKeypoints = False):
    DATASET_DIR = os.path.join(OUTPUT_DIR, "{}/".format(suffix))
    ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations/")
    file_pattern = re.compile('([^/]*)\.([^/.]+)$')
    category_pattern = re.compile('panel', re.IGNORECASE)

    imageId = 0
    annId = 0
    categoryId = 0

    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(DATASET_DIR)
    if ratio not in [0, 1]:
        empty_dir(DATASET_DIR)
    ensure_dir(os.path.join(DATASET_DIR, "val"))
    ensure_dir(os.path.join(DATASET_DIR, "train"))
    ensure_dir(ANNOTATIONS_DIR)

    images_train = []
    images_val = []
    annotations_train = []
    annotations_val = []
    categories = {}
    a = glob(os.path.join(INPUT_DIR, '*.json'.format(suffix)))
    """ Browse through all marked json files """

    if isKeypoints:
        id_keypoints={}

    for file in tqdm(iglob(os.path.join(INPUT_DIR, '*.json'.format(suffix)))):
        imageId += 1
        with open(file, 'r') as f:

            """ Load json files """
            data = json.load(f)
            data_type = data["imagePath"].split('.')[-1]
            """ Separation of train/validation subsets """
            subset = "val" if random.random() < ratio else "train"

            """ Save image file """
            file_name = "{}{}/{:08d}.{}".format(DATASET_DIR, subset, imageId, data_type)
            # print(file_name, file)
            image_data = base64.b64decode(data["imageData"])
            with open(file_name, 'wb') as fi:
                fi.write(image_data)

            """ Get image width x height """
            im = Image.open(file_name)
            (width, height) = im.size

            """ Save image data to index """
            image_obj = {
                'id': imageId,
                'file_name': "{:08d}.{}".format(imageId, data_type),
                'width': width,
                'height': height,
            }

            if subset == "val":
                images_val.append(image_obj)
            else:
                images_train.append(image_obj)

            """ Process each shape (annotation) """
            for shape in data['shapes']:
                annId += 1
                cat = shape['label']

                """ Build category index """
                if cat not in categories:
                    categoryId += 1
                    categories[cat] = {
                        'id': categoryId,
                        'name': cat,
                        'supercategory': 'solar panel' if category_pattern.search(cat) else 'defect'
                    }
                    if isKeypoints:
                        categories[cat]['keypoints']=[]
                        for i in range(len(shape['points'])):
                            categories[cat]['keypoints'].append(str(i+1))
                        categories[cat]['skeleton']=[]
                        for i in range(len(shape['points'])-1):
                            categories[cat]['skeleton'].append([i+1,i+2])
                        categories[cat]['skeleton'].append([len(shape['points']),1])

                category = categories[cat]

                """ Form segment out of points """
                segment = []
                if (len(shape['points']) < 3):
                    print("Error#########################:{}".format(file))

                for [x, y] in shape['points']:
                    segment.append(x)
                    segment.append(y)

                bbox = get_bbox(shape['points'])
                if isKeypoints:
                    bbox[0]=max(0,bbox[0]-1)
                    bbox[1]=max(0,bbox[1]-1)
                    bbox[2]+=2
                    bbox[3]+=2
                [_, _, width, height] = bbox
                """ Add annotations """
                annotation_obj = {
                    'id': annId,
                    'image_id': imageId,
                    'category_id': category['id'],
                    'segmentation': [segment],
                    'bbox': bbox,
                    'area': width * height,
                    'iscrowd': 0,
                }

                if isKeypoints:
                    if not hasattr(id_keypoints,str(category['id'])):
                        id_keypoints[str(category['id'])]=len(shape['points'])
                    else:
                        if id_keypoints[str(category['id'])]!=len(shape['points']):
                            raise ValueError(
                                "The same category must have the same num of keypoints"
                            )

                    keypoints=[]
                    for [x,y] in shape['points']:
                        keypoints.append(x)
                        keypoints.append(y)
                        keypoints.append(2)
                    annotation_obj['keypoints']=keypoints

                    annotation_obj['num_keypoints']=len(shape['points'])

                if subset == "val":
                    annotations_val.append(annotation_obj)
                else:
                    annotations_train.append(annotation_obj)


    with open(ANNOTATIONS_DIR + 'instances_{}_val.json'.format(suffix), 'w') as fa:
        json.dump({
            'images': images_val,
            'annotations': annotations_val,
            'categories': list(categories.values())
        }, fa, indent='  ')

    with open(ANNOTATIONS_DIR + 'instances_{}_train.json'.format(suffix), 'w') as fa:
        json.dump({
            'images': images_train,
            'annotations': annotations_train,
            'categories': list(categories.values())
        }, fa, indent='  ')
    return os.path.join(DATASET_DIR, "train"), os.path.join(DATASET_DIR, "val"), \
           os.path.join(ANNOTATIONS_DIR, 'instances_{}_train.json'.format(suffix)),\
           os.path.join(ANNOTATIONS_DIR, 'instances_{}_val.json'.format(suffix))
