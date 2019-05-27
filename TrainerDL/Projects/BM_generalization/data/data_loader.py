import numpy as np
import glob
import os
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from torchvision import transforms
import torch
from torchvision.datasets import ImageFolder
labels = ["BM", "ZC"]
# 0: bm 1: zc


def build_iaa():
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)
    seq = iaa.Sequential(
        iaa.SomeOf(3,
                   [
                       sometimes(iaa.Superpixels(p_replace=(0, 0.4), n_segments=(200, 400))),
                       iaa.OneOf([
                         iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                         iaa.AverageBlur(k=(2, 5)),  # blur image using local means with kernel sizes between 2 and 7
                         iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                       iaa.Add((-10, 10), per_channel=0.5),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                       iaa.AddToHueAndSaturation((-20, 20)),
                       iaa.Grayscale(alpha=(0.0, 0.5)),
                       iaa.Fog()
                   ],
                   random_order=True
                   )
    )
    return seq


def build_transform(args):
    test_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    train_transform = transforms.Compose([])

    if args.excessive_iaa:
        aug = build_iaa()
        train_transform = transforms.Compose([
            transforms.Lambda(lambda x: np.array(x, dtype=np.uint8)),
            transforms.Lambda(lambda x: aug.augment_image(x)),
            transforms.ToPILImage()
        ])

    train_transform = transforms.Compose([
        train_transform,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=args.brightness_eps,
                               contrast=args.constrast_eps,
                               saturation=args.saturation_eps,
                               hue=args.hue_eps)
    ])

    if args.AffineApply:
        train_transform = transforms.Compose([
            train_transform,
            transforms.RandomApply([transforms.RandomAffine(degrees=args.degrees,
                                                            translate=args.translate,
                                                            scale=args.scale,
                                                            shear=args.sheer)], p=0.3 if args.AffineApply else 0)
        ])

    train_transform = transforms.Compose([
        train_transform,
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


def data_loader(args, is_Train=True):
    splits = ["train", "test", "valid"]
    drop_last_batch = {"train": True, "test": False, "valid": False}
    shuffle = {"train": True, "test": False, "valid": False}
    batchsize = {"train": args.batchsize * 2 if args.mix_up else args.batchsize, "test": args.batchsize, "valid": args.batchsize}

    train_transform, test_transform = build_transform(args)

    dataset = {
        "train": ImageFolder(args.train_img_dir, train_transform if is_Train else test_transform),
        "test": ImageFolder(args.test_img_dir, test_transform),
        "valid": ImageFolder(args.val_img_dir, test_transform),
    }

    if args.mix_up:
        dataset["train"].imgs *= 2
    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=batchsize[x],
                                                 shuffle=shuffle[x],
                                                 num_workers=int(args.num_workers),
                                                 drop_last=drop_last_batch[x]) for x in splits}

    return dataloader

