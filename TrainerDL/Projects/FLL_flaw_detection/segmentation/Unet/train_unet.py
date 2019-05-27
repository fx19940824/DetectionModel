import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import cv2
from tqdm import tqdm
from torch import optim
from argparse import Namespace
from Utils.parsers import parse_cfg
from Projects.FLL_flaw_detection.segmentation.Unet.eval import eval_net
from Projects.FLL_flaw_detection.segmentation.Unet.unet import UNet
from Projects.FLL_flaw_detection.segmentation.Unet.utils import gen_train_val, get_imgs_and_masks, batch


def train_unet(cfg):
    cfgs = parse_cfg(cfg)
    cfgs.update(parse_cfg(cfgs["cfgpath"]))
    cfgs = Namespace(**cfgs)
    net = UNet(n_channels=3, n_classes=cfgs.classes-1).cuda()
    # net.load_state_dict(torch.load("/home/cobot/caid2.0/python/Main/TrainerDL/weights/CP40.pth"))
    iddataset = gen_train_val(cfgs.train_dir, cfgs.val_dir)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(cfgs.epochs, cfgs.batchsize, cfgs.lr, len(iddataset['train']),
               len(iddataset['val']), str(cfgs.weight_out), str(cfgs.gpus)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=cfgs.lr,
                          momentum=0.9,
                          weight_decay=0.0005,
                          )
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0, last_epoch=-1)
    criterion = nn.BCELoss(reduce=False)

    for epoch in range(cfgs.epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, cfgs.epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], cfgs.train_dir, cfgs.img_size)
        val = get_imgs_and_masks(iddataset['val'], cfgs.val_dir, cfgs.img_size, aug=False)

        epoch_loss = 0

        for i, b in tqdm(enumerate(batch(train, cfgs.batchsize)), leave=False, total=N_train // cfgs.batchsize):

            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])
            img_thresholds = np.zeros(true_masks.shape, dtype=np.float32)
            # weighted loss
            # for j, (img,mask) in enumerate(zip(imgs, true_masks)):
            #     img_threshold = cv2.cvtColor((img*255).astype(np.uint8).transpose([1, 2, 0]), cv2.COLOR_RGB2GRAY)
            #     img_threshold[img_threshold > 10] = 255
            #     img_threshold[img_threshold <= 10] = 0
            #     kernel = np.ones((5, 5), np.uint8)
            #     img_threshold = cv2.dilate(img_threshold, kernel, iterations=3)
            #     img_thresholds[j] = img_threshold.astype(np.float32)[np.newaxis, ...] / 255.0 * 19 + 1 + mask * 10

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            img_thresholds = torch.from_numpy(img_thresholds)

            imgs = imgs.cuda()
            true_masks = true_masks.cuda()
            img_thresholds = img_thresholds.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)
            img_thresholds_flat = img_thresholds.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            loss *= img_thresholds_flat
            loss = torch.mean(loss)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if cfgs.weight_out:
            torch.save(net.state_dict(),
                       os.path.join(cfgs.weight_out, 'CP{}.pth'.format(epoch + 1)))
            print('Checkpoint {} saved !'.format(epoch + 1))


if __name__ == '__main__':
    train_unet("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/FLL_flaw_detection/cfgs/unet/train_unet.cfg")
