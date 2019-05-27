#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019 1.4  15:30:27 2019
最终网络文件删减版本
适合处理数据增强完毕的数据
数据增强支持函数:datacreat.py
输入：外部处理好的数据(数据合成,数据增强,随机裁剪等)
@author: cobot wei
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets
import cv2
import os
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import datetime

batch_size = 4
num_epochs = 300
learning_rate = 0.0005
momentum = 0.99
outer_dataset = True  # False, True;

####读取
img_path = '/home/yong/桌面/mytrain/src'
label_path = '/home/yong/桌面/mytrain/label'
temp_path = '/home/yong/桌面/mytrain/temp'


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.scalar = torch.FloatTensor([0]).cuda()

    def forward(self, input, label, p=1.5):
        # label
        # label的形式是[batchsize,w,h]三维
        # label_temp = label + torch.zeros([1, 3, 1, 1], dtype=torch.long).cuda()
        label_temp = label.view(label.size(0), 1, label.size(1), label.size(2))
        label_temp = label_temp.repeat(1, 3, 1, 1)
        target = torch.empty_like(label_temp)
        target[:, 0, :, :] = label_temp[:, 0, :, :] == 0
        target[:, 1, :, :] = label_temp[:, 1, :, :] == 1
        target[:, 2, :, :] = label_temp[:, 2, :, :] == 2
        target = (2 * target - 1).float()
        # weight
        weight = torch.empty_like(label_temp)
        weight[label_temp == 0] = 1.00  # 0.02
        weight[label_temp == 1] = 50.00  # 1.00
        weight[label_temp == 2] = 0.00  # 0.00
        weight = weight.float()

        # input and target are [N,C,H,W]
        raw_loss = torch.max(1.0 - input * target, self.scalar.expand_as(target)).pow(p) * weight
        loss = torch.mean(raw_loss)
        return loss


# conv+bn+PRElu 打包几个torch常用层 方便后面调用
def conv_bn_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.PReLU())
    return layer


# conv+PRElu  去掉batchnorm层
def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.PReLU())
    return layer


##goolenet   定义googlenet的基础结构
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


# 定义网络
class ournet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(ournet, self).__init__()
        self.verbose = verbose  # 是否打印网络信息
        self.block0 = conv_relu(in_channel, out_channels=8, kernel=20, stride=4)
        # 6channels 48x48 512x512
        self.block1 = nn.Sequential(
            conv_bn_relu(8, out_channels=32, kernel=3),
            nn.MaxPool2d(3, 2))
        ##32channels 23x23 255x255

        self.block2 = nn.Sequential(
            inception(32, 8, 8, 16, 8, 16, 8),  # 8+16+16+8
            inception(48, 8, 12, 24, 12, 24, 8),  # 8+24+24+8
            inception(64, 16, 24, 32, 24, 32, 16))  # 16+32+32+16

        ###96channels 23x23
        self.block3 = nn.Sequential(
            conv_bn_relu(96, out_channels=128, kernel=3),
            nn.MaxPool2d(3, 2)
        )

        ###128channels 10x10
        self.block4 = nn.Sequential(
            conv_bn_relu(128, 192, kernel=5),
            # 6x6
            conv_relu(192, 256, kernel=5),
            # 1x1
            conv_relu(256, 128, kernel=3, padding=1),
            # 1x1
            conv_relu(128, 64, kernel=1),
            # 1x1
        )
        self.block5 = nn.Conv2d(64, 3, kernel_size=1)

    #     self.concat = np.concatenate((img,temp),axis=0)

    def forward(self, x1, x2, temp):
        #     x = self.concat(x,temp)
        ###   如果后续需要新增层  只要添加新的block即可
        x = torch.cat([x1, x2], 1)
        x = torch.cat([x, temp], 1)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x


net = ournet(9, 3)

# torch.save(net, 'mynet.pkl')

# # 将网络保存成C++可调用格式,利用tracing转化
net.cuda()
net.eval()
#
example = torch.rand(1, 3, 1200, 1200).cuda()
temp = torch.rand(1, 3, 1200, 1200).cuda()
trace_script_module = torch.jit.trace(net, (example, example, temp))
#
trace_script_module.save('./cnn1.pt')
print(net(example, example, temp))

net2 = torch.jit.load("./cnn1.pt")
net2.eval().cuda()
print(net2(example, example, temp))

# # opencv作为默认读取方式
# def default_loader(path):
#     return cv2.imread(path)
#
#
# # 获取一个文件夹的所有图片
# def get_all_files(path):
#     img_list = []
#     for i in range(len(os.listdir(path))):
#         img_list = img_list + [os.path.join(path, os.listdir(path)[i])]
#     return img_list
#
#
# # ###定义自己的数据集读取，利用torch中的Dataset类来建立
# # ##假设我们的label是图片的形式,这种读取方式适用于lable,temp,image一一对应的情况，即数据增强在外部进行完毕
# # temp也是一个文件夹 裁剪的跟image一一对应
# class our_dataset(Dataset):
#     def __init__(self, img_path, temp_path, label_path, img_transform=None, loader=default_loader):
#         self.img_list = get_all_files(img_path)
#         self.temp_list = get_all_files(temp_path)
#         self.label_list = get_all_files(label_path)
#         self.img_transform = img_transform
#         self.loader = loader
#
#     def __getitem__(self, index):
#         img_path = self.img_list[index]
#         label_path = self.label_list[index]
#         temp_path = self.temp_list[index]
#         # temp_path = self.temp_list[0]          #若temp为一张图片的话  注释上面 用此行
#         img = self.loader(img_path)
#         label = self.loader(label_path)
#         temp = self.loader(temp_path)
#         # label = cv2.imread(label_path,0)
#         # img = img_path
#         if self.img_transform is not None:
#             img = self.img_transform(img)
#             # label = self.img_transform(label)
#             label = trans(label)
#             label = label[:, 102:-101:16, 102:-101:16]  # 根据感受野调整
#             temp = self.img_transform(temp)
#         return img, label, temp
#
#     def __len__(self):
#         return len(self.label_list)
#
#
# ####准确率判断函数  方便监视训练过程　
# def get_acc(output, label):
#     total = output.shape[2] * output.shape[3]
#     _, pred_label = output.max(1)
#     num_correct = (pred_label.long() == label.long()).sum().item()
#     return num_correct / (total * batch_size)
#
#
# #######label的转化方式,因为用默认的转化方式的化会进行归一化,不适合我们的hingloss
# def trans(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = torch.from_numpy(img.transpose((2, 0, 1)))
#
#     return img.long()
#
#
# # 训练图片的预处理
# ####开始训练
# def train(net, train_data, valid_data, num_epochs, optimizer, criterion, mode='original'):
#     if torch.cuda.is_available():
#         net = net.cuda()
#     prev_time = datetime.datetime.now()
#     for epoch in range(num_epochs):
#         train_loss = 0
#         train_acc = 0
#         net = net.train()
#         # net = net.half()
#         for im, label, temp in train_data:
#             # if mode == 'augment':
#             #     im, label = data_augment(im, label)
#             if torch.cuda.is_available():
#                 im = Variable(im.cuda())
#                 label = Variable(label.cuda())
#                 temp = Variable(temp.cuda())
#             else:
#                 im = Variable(im)
#                 label = Variable(label)
#                 temp = Variable(temp)
#
#             ##是否进行数据增强
#
#             label1 = label[:, 0, :, :]
#             # label = label1.view([label.shape[0],1,label.shape[2],label.shape[3]]).long()
#             ##forward
#             output = net(im, temp)
#             loss = criterion(output, label1)
#             ##backward
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             #      train_loss += loss.data[0]
#             train_loss += loss.item()
#             train_acc += get_acc(output, label1)
#
#         # compute the time
#         cur_time = datetime.datetime.now()
#         epoch_str = 'Epoch %d. total_loss%f,train_accuracy%f' % (
#             epoch, train_loss / len(train_data), train_acc / len(train_data))
#
#         # 验证集
#         if valid_data is not None:
#             valid_loss = 0
#             valid_acc = 0
#             net = net.eval()
#             for im, label, temp in valid_data:
#                 if torch.cuda.is_available():
#                     im = Variable(im.cuda(), volatile=True)
#                     label = Variable(label.cuda(), volatile=True)
#                     temp = Variable(temp.cuda(), volatile=True)
#                 else:
#                     im = Variable(im, volatile=True)
#                     label = Variable(label, volatile=True)
#                     temp = Variable(temp, volatile=True)
#                 output = net(im, temp)
#                 label1 = label[:, 0, :, :]
#                 label = label1.view([label.shape[0], 1, label.shape[2], label.shape[3]])
#                 loss = criterion(output, label)
#                 valid_loss += loss.item()
#                 valid_acc += get_acc(output, label)
#             epoch_str = 'Epoch %d. total_loss%f,train_accuracy%f,valid_loss%f,valid_acc%f' % (
#                 epoch, train_loss / len(train_data),
#                 train_acc / len(train_data), valid_loss / len(valid_data),
#                 valid_acc / len(valid_data))
#         cur_time = datetime.datetime.now()
#         h, remainder = divmod((cur_time - prev_time).seconds, 3600)
#         m, s = divmod(remainder, 60)
#         time_str = 'Time%02d:%02d:%02d' % (h, m, s)
#         print(epoch_str + time_str)
#         if epoch % 50 == 0:
#             example = torch.rand(1, 3, 2400, 1200).cuda()  # 有可能需要根据感受野调整
#             temp = torch.rand(1, 3, 2400, 1200).cuda()
#             trace_script_module = torch.jit.trace(net, (example, temp))
#             trace_script_module.save('./cnn_new1.pt')


# net = ournet(6, 3)
#
# # torch.save(net, 'mynet.pkl')
#
# # # 将网络保存成C++可调用格式,利用tracing转化
# net.cuda()
# net.eval()
# #
# example = torch.rand(1, 3, 1200, 1200).cuda()
# temp = torch.rand(1, 3, 1200, 1200).cuda()
# trace_script_module = torch.jit.trace(net, (example, temp))
# #
# trace_script_module.save('./cnn1.pt')
# print(net(example, temp))
#
# net2 = torch.jit.load("./cnn1.pt")
# net2.eval().cuda()
# print(net2(example, temp))



# #####默认归一化！！！！！
# transforms = transforms.ToTensor()
#
# # 读取我们的训练数据，转化为张量
# train_data = our_dataset(img_path=img_path, label_path=label_path, temp_path=temp_path, img_transform=transforms)
# # 在转换成张量之前做数
#
# # train_data = our_dataset(img_path=img_path, label_path=label_path, temp_path=temp_path, img_transform=trans_and_aug)
#
# # 我们分出一部分作为验证集
# # train_set, val_set = get_train_val(train_data)
#
# ###利用dataloader函数分批次的导入数据
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# # test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
#
# ####input 6 channels  output 2 class
# ##将我们的额网络实例化
# net = ournet(6, 3, verbose=False)
#
# # lr = 0.005
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
#
# # criterion = CrossEntropyLoss2d()
# criterion = HingeLoss()
# # criterion = nn.MultiLabelMarginLoss()
#
# # # # # # #start to train
# train(net, train_data=train_loader, valid_data=None, num_epochs=num_epochs, optimizer=optimizer,
#       criterion=criterion)
#
# #########训练完毕后最后再保存一次
# net.cuda()
# net.eval()
#
# example = torch.rand(1, 3, 2400, 1200).cuda()
# temp = torch.rand(1, 3, 2400, 1200).cuda()
# trace_script_module = torch.jit.trace(net, (example, temp))
#
# trace_script_module.save('./cnn_new.pt')
