#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019 1.4  15:30:27 2019
最终网络文件删减版本
适合处理数据增强完毕的数据
数据增强支持函数:datacreat.py
输入：外部处理好的数据(数据合成,数据增强,随机裁剪等)
@author: cobot wei
Modified By Yong @ 20190228
"""
# TODO 优化器的选择优化

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import os
import time
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import datetime
import io
from torch.jit import ScriptModule, script_method, trace
import matplotlib.pyplot as plt

# 训练的基本设置
GPU_NUM = 0  # 指定训练GPU

batch_size = 1
num_epochs = 500
learning_rate = 0.001 #0.002
momentum = 0.8

torch.cuda.set_device(1)

# plt.ion()
# outer_dataset = True  # False, True;

# 图像数据路径设置
left_image_dir = "/media/cobot/8e505336-96a8-4c09-bc62-9ca728a68af3/mobile_washer_train_data/left"
right_image_dir = "/media/cobot/8e505336-96a8-4c09-bc62-9ca728a68af3/mobile_washer_train_data/right"
label_dir = "/media/cobot/8e505336-96a8-4c09-bc62-9ca728a68af3/mobile_washer_train_data/label"

# left_image_dir = "/home/cobot/mobile_data/new_train/left"
# right_image_dir = "/home/cobot/mobile_data/new_train/right"
# label_dir = "/home/cobot/mobile_data/new_train/label"

temp_path = "/home/cobot/mobile_data/templates/templates.bmp"


# -Step.1 网路定义--------------------------------------------------------------------------------------------------------
# -Step.1 (a) 基本模块
# conv+bn+PRElu 打包几个torch常用层 方便后面调用
def conv_bn_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
                          nn.BatchNorm2d(out_channels, eps=1e-3), nn.PReLU())
    return layer


# conv+PRElu  去掉batchnorm层
def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding), nn.PReLU())
    return layer


# GooLeNet   定义基础结构
class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)
        self.branch3x3 = nn.Sequential(conv_relu(in_channel, out2_1, 1),  # 1x1 + 3x3加速
                                       conv_relu(out2_1, out2_3, 3, padding=1))
        self.branch5x5 = nn.Sequential(conv_relu(in_channel, out3_1, 1),  # 1x1 + 5x5加速
                                       conv_relu(out3_1, out3_5, 5, padding=2))
        self.branch_pool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                         conv_relu(in_channel, out4_1, 1))

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


# -Step.1 (b) Loss设计
class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.scalar = torch.FloatTensor([0.0]).cuda()

    def forward(self, inpt, label, p=1.5):
        label_temp = label.view(label.size(0), 1, label.size(1), label.size(2))
        label_temp = label_temp.repeat(1, 3, 1, 1)
        # target = torch.empty_like(label_temp)
        target = torch.zeros_like(label_temp)
        target[:, 0, :, :] = (label_temp[:, 0, :, :] == 0)
        target[:, 1, :, :] = (label_temp[:, 1, :, :] == 1)
        target[:, 2, :, :] = (label_temp[:, 2, :, :] == 2)
        target = (2 * target - 1).float()
        # xxx0 = inpt[0, 0, :, :]
        # xxx0 = xxx0.cpu().detach().numpy()
        # xxx1 = inpt[0, 1, :, :]
        # xxx1 = xxx1.cpu().detach().numpy()
        # xxx2 = inpt[0, 2, :, :]
        # xxx2 = xxx2.cpu().detach().numpy()
        # plt.figure("0")
        # plt.imshow(xxx0)
        # plt.pause(0.1)

        # plt.figure("1")
        # # plt.imshow(xxx1)
        # plt.figure("2")
        # plt.imshow(xxx2)
        # plt.pause(0.1)

        # weight
        weight = torch.empty_like(label_temp)
        weight[label_temp == 0] = 1.00  # 0.02
        weight[label_temp == 1] = 0.00  # 0.00
        weight[label_temp == 2] = 10.00  # 1.00
        weight = weight.float()

        # input and target are [N,C,H,W]
        # raw_loss = torch.max(1.0 - inpt * target, self.scalar.expand_as(target)).pow(p) * weight
        # xx = torch.max(1.0 - inpt * target, self.scalar.expand_as(target))
        raw_loss = torch.max(1.0 - inpt * target, self.scalar.expand_as(inpt)).pow(p) * weight
        # plt.figure("loss0")
        # plt.imshow(raw_loss[0, 0, :, :].cpu().detach().numpy())
        # plt.pause(0.1)

        # plt.figure("label_temp")
        # plt.plot(label_temp[0, 0, 50, :].cpu().detach().numpy())
        # plt.pause(0.01)
        #
        # plt.figure("loss00")
        # plt.plot(raw_loss[0, 0, 50, :].cpu().detach().numpy())
        # plt.pause(0.01)

        # plt.figure("loss2")
        # plt.imshow(raw_loss[0, 2, :, :].cpu().detach().numpy())
        # plt.pause(0.1)

        # plt.figure("loss22")
        # plt.plot(raw_loss[0, 2, 50, :].cpu().detach().numpy())
        # # # plt.show()
        # plt.pause(0.01)
        # plt.show()

        # loss = torch.mean(raw_loss[:, 0, :, :]) + torch.mean(raw_loss[:, 2, :, :])
        loss = torch.mean(raw_loss[:])
        return loss


# -Step.1 (c) 网络主体结构
class WasherNet9(torch.jit.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(WasherNet9, self).__init__()
        self.verbose = verbose  # 是否打印网络信息
        self.block0 = conv_relu(in_channel, out_channels=9, kernel=20, stride=4)  # 感受野58*4+19 =258

        # 6channels 48x48 512x512
        self.block1 = nn.Sequential(
            conv_bn_relu(9, out_channels=32, kernel=3),  # 感受野是 56+2 = 58
            # nn.MaxPool2d(3, 2))  # 感受野是 28×2 = 56
            nn.AvgPool2d(3, 2))  # 感受野是 28×2 = 56
        # #32channels 23x23 255x255

        self.block2 = nn.Sequential(  # 感受野是 24+4 =28
            inception(32, 8, 8, 12, 8, 12, 8),  # 8+16+16+8
            inception(40, 8, 12, 24, 12, 24, 8),  # 8+24+24+8
            inception(64, 16, 24, 32, 24, 32, 16))  # 16+32+32+16

        # ##96channels 23x23
        self.block3 = nn.Sequential(
            conv_bn_relu(96, out_channels=128, kernel=3),  # 感受野是 22+2 =24
            nn.MaxPool2d(3, 2))  # 感受野×2                         # 感受野是 11*2 =22

        # ##128channels 10x10
        self.block4 = nn.Sequential(
            conv_bn_relu(128, 192, kernel=5),  # 6x6               # 感受野是5,7+4 =11
            conv_relu(192, 256, kernel=5),  # 1x1                  # 感受野是5，3+4=7
            conv_bn_relu(256, 128, kernel=3, padding=1),  # 1x1       # 感受野是3
            conv_relu(128, 64, kernel=1), )  # 1x1                 # 感受野是1

        self.block5 = nn.Conv2d(64, 3, kernel_size=1)  # 感受野是1

    def forward(self, x1, x2, temp):
        x = torch.cat((x1, x2), 1)  # 构成需要的通道数量
        x = torch.cat((x, temp), 1)  # 构成需要的通道数量
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


# -Step.2 数据处理--------------------------------------------------------------------------------------------------------
# -Step.2 (a) 基础函数
def default_loader(path):  # OpenCV作为默认图像读取方式
    # print(path)
    return cv2.imread(path)


def image_loader(path):  # OpenCV作为默认图像读取方式
    # print(path)
    return cv2.imread(path, -1)


def label_loader(path):  # OpenCV作为默认图像读取方式
    # print(path)
    return cv2.imread(path, 0)


def get_all_files(path):  # 获取文件夹下的图片绝对路径
    img_list = []
    for i in range(len(os.listdir(path))):
        img_list = img_list + [os.path.join(path, os.listdir(path)[i])]
    return img_list


def label_trans(label):  # label的转化方式,因为用默认的转化方式的化会进行归一化,不适合我们的hingloss
    # label = label[:, :, torch.newaxis] // 122
    # label = label.unsqueeze(0) // 122
    label = label // 122
    # label = torch.from_numpy(label.transpose((2, 0, 1)))
    # label = label.transpose((2, 0, 1))
    return label.long()


# torch.unsqueeze()


def image_trans(image):  # label的转化方式,因为用默认的转化方式的化会进行归一化,不适合我们的hingloss
    # image = image.astype(torch.float32) / torch.float32(255)
    image = image / 255.0
    # image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = image.transpose(0, 2)
    image = image.transpose(1, 2)

    # image = image.transpose(2, 0, 1)
    return image


# 定义自己的数据集读取，利用torch中的Dataset类来建立
# 假设我们的label是图片的形式,这种读取方式适用于label,temp,image一一对应的情况，即数据增强在外部进行完毕
# temp也是一个文件夹 裁剪的跟image一一对应
class WasherData(Dataset):
    def __init__(self, left_image_dir, right_image_dir, temp_path, label_dir, img_transform=None,
                 loader=default_loader):
        self.img_list1 = get_all_files(left_image_dir)
        self.img_list2 = get_all_files(right_image_dir)
        self.temp_path = temp_path
        self.label_list = get_all_files(label_dir)
        self.img_transform = img_transform  # 定义预处理
        self.loader = loader

        temp = self.loader(self.temp_path)
        self.temp = self.img_transform(temp)

    def __getitem__(self, index):
        l_image_dir = self.img_list1[index]
        r_image_dir = self.img_list2[index]
        l_dir = self.label_list[index]
        img1 = torch.from_numpy(image_loader(l_image_dir).astype(np.float32)).cuda()
        img2 = torch.from_numpy(image_loader(r_image_dir).astype(np.float32)).cuda()
        label = torch.from_numpy(label_loader(l_dir).astype(np.long)).cuda()

        # label = cv2.imread(label_dir,0)
        # img = img_path
        if self.img_transform is not None:
            # img1 = self.img_transform(img1)  # TODO 看下操作
            # img2 = self.img_transform(img2)
            img1 = image_trans(img1)
            img2 = image_trans(img2)
            label = label_trans(label)
            # label = label[100:-100:16, 98:-97:16]  # 根据感受野调整
            label = label[100:-100:16, 96:-95:16]  # 根据感受野调整
        return img1, img2, label, self.temp

    def __len__(self):
        return len(self.label_list)


# -Step.3 网络训练--------------------------------------------------------------------------------------------------------
# -Step.3 (a) 准确率函数，监视训练过程　
def get_acc(output, label):
    total = torch.nonzero(label == 0).shape[0] + torch.nonzero(label == 2).shape[0]
    # total = output.shape[0] * output.shape[2] * output.shape[3]  # 正常的输出点位
    # _, pred_label = output.max(1)
    pred_label = output.argmax(1)
    num_correct = (pred_label.long() == label.long()).sum().item()
    return num_correct / total


# -Step.3 (b) 训练函数，封装训练过程　
def train(net, train_data, valid_data, num_epochs, optimizer, criterion, mode='original'):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()  # 避免验证集的问题
        # net = net.eval()
        for im1, im2, label, temp in train_data:
            # if mode == 'augment':
            #     im, label = data_augment(im, label)
            if torch.cuda.is_available():
                # start_time = time.time()
                im1 = Variable(im1.cuda())  # 0.15s
                im2 = Variable(im2.cuda())
                label = Variable(label.cuda())
                temp = Variable(temp.cuda())
                # im1 = im1.cuda()
                # im2 = im2.cuda()
                # label = label.cuda()
                # temp = temp.cuda()

            else:
                im1 = Variable(im1)
                im2 = Variable(im2)
                label = Variable(label)
                temp = Variable(temp)

            # label1 = label[:, 0, :, :]
            # label = label1.view([label.shape[0],1,label.shape[2],label.shape[3]]).long()
            # #forward
            output = net(im1, im2, temp)  # 0.12s
            loss = criterion(output, label)

            # #backward
            optimizer.zero_grad()  # 0.3s
            loss.backward()
            optimizer.step()
            #      train_loss += loss.data[0]
            # train_loss += loss.item()
            train_loss += torch.mean(loss)

            print(get_acc(output, label))
            train_acc += get_acc(output, label)  # 0.0003s

        # # compute the time
        # cur_time = datetime.datetime.now()
        epoch_str = 'Epoch: %d, total_loss:%8f, train_accuracy:%8f' % (
            epoch, train_loss / (len(train_data) / batch_size), train_acc / (len(train_data)) / batch_size)
        torch.save(net, "temp.pth")
        # if train_loss / (len(train_data) / batch_size)<0.001:


        # # 验证集
        # if valid_data is not None:
        #     valid_loss = 0
        #     valid_acc = 0
        #     net = net.eval()
        #     for im1, im2, label, temp in valid_data:
        #         if torch.cuda.is_available():
        #             im1 = Variable(im1.cuda(), volatile=True)
        #             im2 = Variable(im2.cuda())
        #             label = Variable(label.cuda(), volatile=True)
        #             temp = Variable(temp.cuda(), volatile=True)
        #         else:
        #             im1 = Variable(im1, volatile=True)
        #             im2 = Variable(im2, volatile=True)
        #             label = Variable(label, volatile=True)
        #             temp = Variable(temp, volatile=True)
        #         output = net(im1, im2, temp)
        #         label1 = label[:, 0, :, :]
        #         label = label1.view([label.shape[0], 1, label.shape[2], label.shape[3]])
        #         loss = criterion(output, label)
        #         valid_loss += loss.item()
        #         valid_acc += get_acc(output, label)
        #     epoch_str = 'Epoch %d. total_loss%f,train_accuracy%f,valid_loss%f,valid_acc%f' % (
        #         epoch, train_loss / len(train_data),
        #         train_acc / len(train_data), valid_loss / len(valid_data),
        #         valid_acc / len(valid_data))

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = ', Time:%02d:%02d:%02d' % (h, m, s)
        print(epoch_str + time_str)
        if (epoch + 1) % 1 == 0:
            example1 = torch.rand(1, 3, 2400, 1200).cuda()
            example2 = torch.rand(1, 3, 2400, 1200).cuda()
            temp = torch.rand(1, 3, 2400, 1200).cuda()
            trace_script_module = torch.jit.trace(net, (example1, example2, temp))
            trace_script_module.save("./washer_model/washer_model" + str(epoch) + ".pt")
            # trace_script_module.module.save("./washer_model/washer_model" + str(epoch) + ".pt")  # 并行GPU的保存方法


# -Step.4 程序主函数--------------------------------------------------------------------------------------------------------
def out_train():
    # -得到转换函数
    transform = transforms.ToTensor()

    # -加载数据loader
    train_data = WasherData(left_image_dir=left_image_dir, right_image_dir=right_image_dir, label_dir=label_dir,
                           temp_path=temp_path,
                           img_transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # 利用loader函数分批次的导入数据

    # -将网络实例化(input 9 channels  output 3 class)

    if os.path.exists("temp.pth"):
        net = torch.load("temp.pth")
    else:
        net = WasherNet9(9, 3, verbose=True)
    # net = WasherNet9(9, 3, verbose=True)

    # print(net)
    # net.cuda()
    # net = torch.nn.DataParallel(net.cuda(), device_ids=[0, 1, 2])  # 设置并行GPU训练模式(6通道还没修改)

    # # -李勇新增的测试感受野的小代码
    # net.cuda()
    # net.eval()
    # size_x = 7720
    # size_y = 1000
    # left_image = torch.rand(1, 3, size_x, size_y).cuda()
    # right_image = torch.rand(1, 3, size_x, size_y).cuda()
    # temp = torch.rand(1, 3, size_x, size_y).cuda()
    # output = net(left_image, right_image, temp)
    # print(output.data.shape)
    # # TODO 设置并行GPU训练模式

    # -指定训练loss和优化函数
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, nesterov=False)
    criterion = HingeLoss()
    # print(dir(optimizer))
    # criterion = CrossEntropyLoss2d()
    # criterion = nn.MultiLabelMarginLoss()

    # -开始训练
    train(net, train_data=train_loader, valid_data=None, num_epochs=num_epochs, optimizer=optimizer,
          criterion=criterion)

    plt.show()
    # 训练完毕后最后再保存一次
    with torch.no_grad():
        net.cuda()
        net.eval()
        left_image = torch.rand(1, 3, 2400, 1200).cuda()
        right_image = torch.rand(1, 3, 2400, 1200).cuda()
        temp = torch.rand(1, 3, 2400, 1200).cuda()
        trace_script_module = torch.jit.trace(net, (left_image, right_image, temp))

        trace_script_module.save("./washer_model/WasherNet.pt")


if __name__ == "__main__":
    out_train()
