import json
import os
import glob
import base64
from PIL import Image
import argparse
from tqdm import tqdm
import cv2

FLAGS = None

#instance json
'''
images={
    'id':0,
    'file_name':"",
    'width':0,
    'height':0,
}

annotations={
    'id':0,
    'image_id': 0,
    'category_id': 0,
    'segmentation': 0,
    'bbox': 0,
    'area': 0,
    'iscrowd': 0,
}
'''

def main():
    if not os.path.exists(os.path.join(FLAGS.dir_output,FLAGS.subset)):
        os.makedirs(os.path.join(FLAGS.dir_output,FLAGS.subset))

    img_list = glob.glob(FLAGS.dir_img+'*.png')

    images_output=[]
    annotations_output=[]
    categories={
        str(FLAGS.objname):
        {
            'id':1,
            'name':FLAGS.objname,
            'supercategory':FLAGS.objname
        }
    }
    imageId = 1
    annId = 1
    for img_path in tqdm(img_list):
        label_path = img_path.replace(os.path.split(os.path.split(FLAGS.dir_img)[0])[-1],os.path.split(os.path.split(FLAGS.dir_label)[0])[-1])
        if not os.path.exists(label_path):
            continue
        
        #get shapes
        img = cv2.imread(label_path, 0)
        width,height=img.shape
        cnt,_ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        #skip if there is no obj
        if len(cnt)==0:
            continue

        annotation_obj=None
        for points in cnt:

            epsilon=0.01*cv2.arcLength(points,True)
            points=cv2.approxPolyDP(points,epsilon,True)

            isEdge=False
            segment=[]
            min_x = None
            min_y = None
            max_x = None
            max_y = None
            for [[x,y]] in points:
                if x==0 or y==0 or x==width-1 or y==height-1:
                    isEdge=True
                    break
                x=int(x)
                y=int(y)
                segment.append(x)
                segment.append(y)
                min_x = x if not min_x else min(x, min_x)
                min_y = y if not min_y else min(y, min_y)
                max_x = x if not max_x else max(x, max_x)
                max_y = y if not max_y else max(y, max_y)
            if isEdge:
                continue
            box_width,box_height = max_x - min_x,max_y - min_y
            bbox = [min_x, min_y, box_width, box_height]

            annotation_obj = {
                'id':annId,
                'image_id': imageId,
                'category_id': 1,
                'segmentation': [segment],
                'bbox': bbox,
                'area': box_width*box_height,
                'iscrowd': 0,
            }
            annotations_output.append(annotation_obj)
            annId+=1

        if annotation_obj is None:
            continue
    
        #get image information
        prefix, img_name=os.path.split(img_path)
        data_type = img_name.split('.')[-1]
        im = Image.open(img_path)
        (width,height) = im.size

        #save image file
        file_name = '{}/{}/{:08d}.{}'.format(FLAGS.dir_output,FLAGS.subset,imageId,'png')
        f = open(img_path, 'rb')
        image_data = base64.b64encode(f.read()).decode('utf-8')
        image_data = base64.b64decode(image_data)
        with open(file_name, 'wb') as fi:
            fi.write(image_data)

        image_obj={
            'id': imageId,
            'file_name': "{:08d}.{}".format(imageId, 'png'),
            'width': width,
            'height': height,
        }
        images_output.append(image_obj)

        imageId+=1

    with open(FLAGS.dir_output+'{}_instances_{}.json'.format(FLAGS.objname,FLAGS.subset),'w') as f:
        json.dump({
            'images':images_output,
            'annotations':annotations_output,
            'categories':list(categories.values())
        },f,indent='  ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir_img',
        type=str,
        default='/home/fangxin/Dataset/building/train/image/',
    )
    parser.add_argument(
        '--dir_label',
        type=str,
        default='/home/fangxin/Dataset/building/train/label/',
    )
    parser.add_argument(
        '--objname',
        type=str,
        default='building'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='train'
    )
    parser.add_argument(
        '--dir_output',
        type=str,
        default='/home/fangxin/Dataset/building/coco/',
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()