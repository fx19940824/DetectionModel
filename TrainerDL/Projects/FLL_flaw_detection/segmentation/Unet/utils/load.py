#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import cv2
from Projects.FLL_flaw_detection.segmentation.Unet.utils.utils import resize_and_crop, get_square, normalize, hwc_to_chw
from Projects.FLL_flaw_detection.segmentation.Unet.augmentation import augmentation

def squeeze(mask):
    if len(mask.shape) ==3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.float32)
    if np.max(mask)==255:
        mask /= 255.0
    return mask[..., np.newaxis]


def to_cropped_imgs(ids, dir, isize):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        im = resize_and_crop(Image.open(os.path.join(dir, id)), isize=isize)
        yield im


def get_imgs_and_masks(ids, dir, isize=448, aug=True):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir, isize)

    # need to transform from HWC to CHW
    if aug:
        imgs = map(augmentation, imgs)
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)
    ids_mask = [name.replace('.', '_mask.') for name in ids]

    masks = to_cropped_imgs(ids_mask, dir, isize)
    masks_squeezed = map(squeeze, masks)
    masks_switched = map(hwc_to_chw, masks_squeezed)
    if aug:
        gen = map(augmentwithmask, zip(imgs_normalized, masks_switched))
    else:
        gen = zip(imgs_normalized, masks_switched)
    return gen


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.png')
    mask = Image.open(dir_mask + id + '.png')
    return np.array(im), np.array(mask)


def augmentwithmask(inputs):
    flip_code = np.random.randint(-1, 3)
    img, mask = inputs
    if flip_code != 2:
        img = cv2.flip(img, flip_code)
        mask = cv2.flip(mask, flip_code)
    return img, mask