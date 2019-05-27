import glob
from imgaug import augmenters as iaa
import cv2
from  torchvision import transforms
import numpy as np
def build_iaa():
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)
    seq = iaa.Sequential(
        iaa.SomeOf(3,
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 0.4), n_segments=(200, 400))),
                       iaa.OneOf([
                         # iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                         # iaa.AverageBlur(k=(2, 5)),  # blur image using local means with kernel sizes between 2 and 7
                         # iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                       iaa.Add((-10, 10), per_channel=0.5),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                       # iaa.ContrastNormalization((0.5, 1.0), per_channel=0.5),
                       # iaa.AddToHueAndSaturation((-20, 20)),
                       # iaa.
                   ],
                   random_order=True
                   )
    )
    return seq

train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1,
                               contrast=0.1,
                               saturation=0.1,
                               hue=0.05),

    ])
tf = build_iaa()
imgs = glob.glob("/home/cobot/FLL/DATA/valid_board/*.png")