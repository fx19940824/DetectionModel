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
batch_size = 1
transform = transforms.ToTensor()  # 增加维度,变成0-1的范围内
torch.cuda.set_device(0)


def get_acc(output, label):
    total = torch.nonzero(label == 0).shape[0] + torch.nonzero(label == 2).shape[0]
    # total = output.shape[0] * output.shape[2] * output.shape[3]  # 正常的输出点位
    # _, pred_label = output.max(1)
    pred_label = output.argmax(1)
    num_correct = (pred_label.long() == label.long()).sum().item()
    return num_correct / total


with torch.no_grad():
    # step.2 加载网络,数据,并做数据转换
    # net = torch.jit.load("./washer_model/ScrewNet.pt")
    net = torch.jit.load("./washer_model/washer_model0.pt")
    device = torch
    # net.cuda().half()
    net.cuda()
    net.eval()
    # print(net)

    bias = cv2.imread(os.path.join(os.getcwd() + "/washer_model/washer_bias.bmp"))
    bias = (bias.astype(np.float32)) * 20.0 / 255.0
    bias = bias[:, :, 0]*1.5
    bias[212, 100] = -100
    bias = torch.from_numpy(bias)
    # bias = bias.transpose(0, 2)
    # bias = bias.transpose(1, 2)
    # xx = np.reshape(bias, [-1])
    # plt.figure()
    # plt.plot(xx)
    # plt.show()
    bias = bias.cuda()
    bias = bias.view([1, 1, bias.size(0), bias.size(1)])

    # left_image_dir = "/home/cobot/Pictures/full/left"
    # right_image_dir = "/home/cobot/Pictures/full/right"
    # label_dir = ""
    # left_image_dir = "/home/cobot/mobile_data/new_train/left"
    # right_image_dir = "/home/cobot/mobile_data/new_train/right"
    # label_dir = "/home/cobot/mobile_data/new_train/label"
    # left_image_dir = "/media/cobot/5C8B2D882D247B56/data/_0normal_images/full/left"
    # right_image_dir = "/media/cobot/5C8B2D882D247B56/data/_0normal_images/full/right"
    # label_dir = ""
    #
    # left_image_dir = "/media/cobot/94584AF0584AD0A2/data/_2screw_patches/第五次标注/多垫片/整/left"
    # right_image_dir = "/media/cobot/94584AF0584AD0A2/data/_2screw_patches/第五次标注/多垫片/整/right"
    # label_dir = ""

    left_image_dir = "/home/cobot/Desktop/temp/left"
    right_image_dir = "/home/cobot/Desktop/temp/right"
    label_dir = ""

    temp = cv2.imread("/home/cobot/mobile_data/templates/templates.bmp")
    temp = transform(temp)
    temp = temp.cuda()
    temp = temp.view([batch_size, temp.size(0), temp.size(1), temp.size(2)])

    for name in os.listdir(left_image_dir):
        left_imagex = cv2.imread(os.path.join(left_image_dir, name))
        right_image = cv2.imread(os.path.join(right_image_dir, name))
        label = cv2.imread(os.path.join(label_dir, name), 0)
        if left_imagex is None or right_image is None:
            print("图像路径" + name + "不存在！")
            continue
        if label is None:
            label = np.zeros_like(right_image)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        left_image = transform(left_imagex)
        right_image = transform(right_image)

        left_image = left_image.cuda()
        right_image = right_image.cuda()

        left_image = left_image.view([batch_size, left_image.size(0), left_image.size(1), left_image.size(2)])
        right_image = right_image.view([batch_size, right_image.size(0), right_image.size(1), right_image.size(2)])

        # left_image = Variable(left_image)
        # right_image = Variable(right_image)
        # temp = Variable(temp)


        # step.3 网络预测，结果解析
        # print(dir(net))
        out = net(x1=left_image, x2=right_image, temp=temp)
        out[:, 0, :, :] = out[:, 0, :, :] + bias
        # print(torch.max(bias[:, 0, :, :]))
        # out = out + bias
        # # out = net.forward(left_image, right_image, temp)
        # plt.figure()
        # data = out.cpu().data.numpy().reshape([3, -1]).transpose()
        # bias = data[:, 2] - data[:, 0]
        # bias[bias < 0] = 0
        # post_data = data.copy()
        # post_data[:, 2] = post_data[:, 2] - bias
        # # np.transpose
        # plt.plot(data[::, ::2])
        # plt.xlim([49500, 50000])
        # plt.figure()
        # plt.plot(post_data[::, ::2])
        # plt.xlim([49500, 50000])
        # plt.show()

        # _, test_label = out.max(1)
        test_label = torch.argmax(out, dim=1)
        test_label = test_label.cpu().data.numpy()  # 转到cpu，转到numpy，转换维度
        test_label = np.squeeze(test_label)
        # test_label = (test_label[0, :, :] < 0) * (test_label[2, :, :] > 0)

        labelx = torch.from_numpy(label.astype(np.long)).cuda()
        labelx = labelx // 122
        labelx = labelx.long()
        labelx = labelx[100:-100:16, 96:-95:16]  # 根据感受野调整

        test_acc = get_acc(out, labelx)  # 0.0003s
        print(test_acc)

        cv2.namedWindow("prediction", cv2.WINDOW_NORMAL)
        cv2.namedWindow("truth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)

        cv2.imshow("prediction", 120 * test_label.astype(np.uint8))
        cv2.imshow("truth", label)
        cv2.imshow("image", left_imagex)
        cv2.waitKey()

        # cv2.imwrite('prediction.png', 255 * result)
        # cv2.imwrite('thetruelabel.png', 122 * test_label1)
        #
        # print(test_label)
        #
        # test_label1 = cv2.imread('/home/yong/桌面/mytrain/label/5.png')
        #
        # test_temp = cv2.imread('/home/yong/桌面/mytrain/temp/5.png')
        # test_temp = transform(test_temp)
        # print(test_temp.size())
        # test_temp = test_temp.cuda()
        # test_temp = test_temp.view([batch_size, test_temp.size(0), test_temp.size(1), test_temp.size(2)])
        # # test_img = cv2.imread(
        # #     '/home/cobot/Desktop/weibeifen/cobot/缺陷图像/washer/case5/images/2018.12.06_22:39:41.089_7f67459cdb40.png')
        # test_img = cv2.imread(
        #     '/home/yong/桌面/mytrain/src/5.png')
        # test_img = transform(test_img)
        # test_img = test_img.cuda()
        # test_img = test_img.view([batch_size, test_img.size(0), test_img.size(1), test_img.size(2)])
        #
        # net = torch.jit.load('cnn_new222222222.pt')

        # test_label = net(test_img, test_img, test_temp)
        #
        # _, test_label = test_label.max(1)
        # print(test_label)
        # result = test_label.cpu().detach().numpy()[0, :, :]
        #
        # print(np.max(result))
        # cv2.imwrite('mypredict111111.png', 255 * result)
        # cv2.imwrite('thetruelabel.png', 122 * test_label1)
