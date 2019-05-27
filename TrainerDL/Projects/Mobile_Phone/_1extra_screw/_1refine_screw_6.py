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
import matplotlib.pyplot as plt

# step.1 基础设置
IMG_NUM = 340  # 更新bias用的图像数量，-1表示全部
SHOW_PLOT = False  # 是否显示中间数据
torch.cuda.set_device(0)

left_image_dir = "/home/cobot/mobile_data/screw_train/20190318/left"
label_dir = ""
#
# left_image_dir = "/home/cobot/Pictures/full/left"
# right_image_dir = "/home/cobot/Pictures/full/right"
# label_dir = ""

temp_path = "/home/cobot/mobile_data/templates/templates.bmp"
net_path = "./screw_model/screw_model0.pt"

if SHOW_PLOT:
    plt.ion()  # interactive mode

transform = transforms.ToTensor()  # 增加维度,变成0-1的范围内

for area in [2, 3, 4]:  # 分区域构建bias
    batch_size = 1

    with torch.no_grad():
        # step.2 加载网络,数据,并做数据转换
        # net = torch.jit.load("./screw_model/ScrewNet.pt")
        net = torch.jit.load(net_path)
        net.cuda().half()
        net.eval()

        temp = cv2.imread(temp_path)
        if area > -1:
            temp = temp[:, (4 - area) * 1550: ((4 - area) * 1550 + 1520), :]
        temp = transform(temp)
        temp = temp.cuda().half()
        temp = temp.view([batch_size, temp.size(0), temp.size(1), temp.size(2)])
        temp = Variable(temp)

        bias_all = []
        if SHOW_PLOT:
            plt.figure()

        names = os.listdir(left_image_dir)
        for k, name in enumerate(names[0:IMG_NUM]):
            print(area, name)
            net.eval().half()
            # net.share_memory()
            # net = torch.nn.DataParallel(net.cuda(), device_ids=[0, 1, 2])  # 设置并行GPU训练模式(6通道还没修改)

            left_imagex = cv2.imread(os.path.join(left_image_dir, name))
            label = cv2.imread(os.path.join(label_dir, name))
            if left_imagex is None:
                print("图像路径" + name + "不存在！")
                continue
            if label is None:
                label = np.zeros_like(left_imagex)
                label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

            if area > -1:
                left_imagex = left_imagex[:, (4 - area) * 1550: ((4 - area) * 1550 + 1520), :]
                label = label[:, (4 - area) * 1550: ((4 - area) * 1550 + 1520)]

            left_image = transform(left_imagex)

            left_image = left_image.cuda().half()

            left_image = left_image.view([batch_size, left_image.size(0), left_image.size(1), left_image.size(2)])

            left_image = Variable(left_image)

            # step.3 网络预测，结果解析
            out = net.forward(left_image, temp)
            bias = out[:, 2, :, :] - out[:, 0, :, :]
            bias[bias < 0] = torch.zeros(1, 1).cuda()

            if k == 0:
                all_flag = False
                bias_all = bias
            bias_all = torch.max(bias, bias_all)

            if SHOW_PLOT:
                data = bias_all.cpu().data.numpy().reshape([1, -1]).transpose()
                plt.plot(data[::, ::2])
                # plt.xlim([49000, 50000])
                plt.pause(0.001)  # pause a bit so that plots are updated

            # 保存数据
            if k % 10 == 0:
                data = np.squeeze(bias_all.cpu().data.numpy()) * 255 / 20
                data = np.clip(data, 0, 255)
                data = data.astype(np.uint8)
                xxx = np.zeros([data.shape[0], data.shape[1], 3], dtype=np.uint8)
                xxx[:, :, 0] = data
                if area > -1:
                    cv2.imwrite("./screw_model/screw_bias_" + str(area) + ".bmp", xxx)
                else:
                    cv2.imwrite("./screw_model/screw_bias.bmp", xxx)

if SHOW_PLOT:
    plt.show()
