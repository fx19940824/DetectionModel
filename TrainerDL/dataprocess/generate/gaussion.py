import glob
import json
import cv2
import sys
import base64
from imgaug import augmenters as iaa

def main(dir_source):
    seq=iaa.Sequential([
        iaa.GaussianBlur((0 ,3.0))
    ])

    imglist = []
    for index, filename in enumerate(glob.glob(dir_source + '/*.png')):
        str_index = str(index)
        theimgname = filename.split('/')[-1]
        theimgname = theimgname.split('.png')[0]
        filepath = filename.split('.png')[0]
        imglist.append((filepath, theimgname))

    thenum = 0
    thetotal = len(imglist)
    for name, imgname in imglist:
        image = cv2.imread(name + ".png", cv2.IMREAD_COLOR)
        images_aug=seq.augment_images(image)
        cv2.imwrite(dir_source + imgname + '_aug.png', images_aug)
        thenum += 1

        sys.stdout.write('{0}/{1}\r'.format(thenum, thetotal))
        sys.stdout.flush()

    print("total:" + str(thetotal))

if __name__=='__main__':
    dir_source = '/home/cobot/newdelete1/test/'

    main(dir_source)