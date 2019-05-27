import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from argparse import Namespace
from Utils.parsers import parse_cfg
from Unet.eval import eval_net
from Unet.unet import UNet
from Unet.utils import gen_train_val, get_imgs_and_masks, batch


def train_unet(cfg):
    cfgs = parse_cfg(cfg)
    cfgs.update(parse_cfg(cfgs["cfgpath"]))
    cfgs = Namespace(**cfgs)
    net = UNet(n_channels=3, n_classes=cfgs.classes-1).cuda()

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
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

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

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
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
    train_unet("/home/cobot/caid2.0/python/Main/TrainerDL/cfgs/unet/train_unet.cfg")
