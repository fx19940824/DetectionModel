#!/usr/bin/python2
# coding:utf-8

"""
@author: Yong Li
@contact: liyong@cobotsys.com
@software: PyCharm
@file: data_set_maker2.py
@time: 18-9-28 下午3:46
协作网络的数据生成:根据网络的预测结果处理
产生lmdb,分别存储6通道输入图像,和多个网络的1通道的标签值
尺寸:输入图像不做裁剪,标签与输出匹配
"""

import numpy as np
import cv2
import os
import random
import lmdb
import sys
import copy
from ctools.basic_func import del_file, image_correct, get_all_files

sys.path.append(r'/home/cobot/cellphone_project/caffe/caffe-master/python')
import caffe

root = "/home/cobot/cellphone_project"


def make_lmdb(template):
    # ---指定各种文件夹-------------------------------------------------------------------------------------------------------
    # 输入图像文件位置
    image_dir = root + "/data/4screw/train_images"
    label_dir = root + "/data/4screw/train_labels"
    # template_path = root + "/data/3templates/2.png"

    image_lmdb = root + "/data/4screw/train_lmdb/image_lmdb"
    label_lmdb1 = root + "/data/4screw/train_lmdb/label1_lmdb"
    label_lmdb2 = root + "/data/4screw/train_lmdb/label2_lmdb"

    # temp_dir = "/media/yong/data/cellphone_project/data/screw/temp"

    del_file(image_lmdb)
    del_file(label_lmdb1)
    del_file(label_lmdb2)

    # 模板的问题
    # template_image = cv2.imread(template_path)
    # template = cv2.imread(template_path)

    # 图像文件的内容
    files = get_all_files(image_dir)
    files = files * 3
    random.shuffle(files)
    # files = files[:100]

    # ---lmdb文件信息 & 启动写入----------------------------------------------------------------------------------------------
    env_image = lmdb.open(image_lmdb, map_size=1e12)
    env_label1 = lmdb.open(label_lmdb1, map_size=1e12)
    env_label2 = lmdb.open(label_lmdb2, map_size=1e12)
    txn_image = env_image.begin(write=True)
    txn_label1 = env_label1.begin(write=True)
    txn_label2 = env_label2.begin(write=True)

    # xx = np.linspace(0, 200, 201)
    # yy = np.linspace(0, 408, 409)
    # zz = 1
    # xx, yy, zz = np.meshgrid(yy, xx, zz)

    for short_file_name, k in zip(files, range(len(files))):
        print("Data set maker:", k, short_file_name)

        # Step 1. 读图 & roi & registration -----------------------------------------------------------------------
        # name_without_ext, ext = os.path.splitext(short_file_name)  # 处理名称 & 读图
        image_name = image_dir + os.sep + short_file_name
        label_name = label_dir + os.sep + short_file_name

        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)

        if image is None:
            print "no image with name", image_name
            continue
        elif label is None:
            print "no label with name", image_name
            continue

        image, M = image_correct(template, image)  # 与模板对齐

        # 使用无损压缩PNG格式,避免图像压缩带来的问题
        label = np.array(label == 255).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))  # 形态学处理下，螺钉有效区域增加下
        label = cv2.morphologyEx(label, cv2.MORPH_DILATE, kernel)

        # 中间模糊地带的数据,通过形态学open操作得到label_open,并与模板匹配变形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))  # 形态学处理下，避免负样本中包含螺钉
        label_open = cv2.morphologyEx(np.array(label == 255).astype(np.uint8), cv2.MORPH_DILATE, kernel) * 255

        label = cv2.warpPerspective(label, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
        label_open = cv2.warpPerspective(label_open, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

        # 生成初始标注数据----------------------------------------------------------------------------------------------------
        label_result = 2 * np.ones(label.shape, dtype=np.uint8)  # 默认值为2,模糊地带不做处理
        label_result[label == 255] = 1  # 有螺钉值为1
        label_result[label_open != 255] = 0  # 没有螺钉,背景值为0

        # 深度拷贝下
        img = copy.copy(image)
        ref = copy.copy(template)

        # 截取ROI,主要是图太大了显存搞不定,训练效果也不好
        steps = np.ceil((np.array(image.shape[:2]) + 128) / 512)
        ind_x = np.round(np.linspace(0, image.shape[0] - 512 - 4, steps[0]))
        ind_y = np.round(np.linspace(0, image.shape[1] - 512 - 4, steps[1]))
        ind_x = np.int(np.random.choice(ind_x) + np.random.randint(0, 4))
        ind_y = np.int(np.random.choice(ind_y) + np.random.randint(0, 4))

        img = img[ind_x:ind_x + 512, ind_y:ind_y + 512, :]
        ref = ref[ind_x:ind_x + 512, ind_y:ind_y + 512, :]
        label_result = label_result[ind_x:ind_x + 512, ind_y:ind_y + 512]

        # 匹配输出标签
        label_result = label_result[24:-23:4, 24:-23:4]
        label_result = label_result[:, :, np.newaxis]

        # cv2.imwrite(os.path.join(temp_dir, short_file_name[:-4]) + str(k) + "img.png", img) # Debug 查看输入lmdb数据是否异常
        # cv2.imwrite(os.path.join(temp_dir, short_file_name[:-4]) + str(k) + "tmp.png", ref)
        # cv2.imwrite(os.path.join(temp_dir, short_file_name[:-4]) + str(k) + "res.png", label_result * 120)

        # 准备成lmdb格式------------------------------------------------------------------------------------------------
        img = np.transpose(img, (2, 0, 1))
        ref = np.transpose(ref, (2, 0, 1))
        X = np.concatenate((img, ref), axis=0)

        label_result1 = copy.copy(label_result)
        label_result2 = copy.copy(label_result)
        # 写入lmdb----------------------------------------------------------------------------------------------------------
        # - 图像写入

        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[0]
        datum.height = X.shape[1]
        datum.width = X.shape[2]
        datum.data = X.tobytes()  # or .tostring() if numpy < 1.9
        str_id = '{:08}'.format(2 * k + 1)
        # The encode is only essential in Python 3
        txn_image.put(str_id.encode('ascii'), datum.SerializeToString())

        # - 标签写入
        X = np.transpose(label_result1, (2, 0, 1))
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[0]
        datum.height = X.shape[1]
        datum.width = X.shape[2]
        datum.data = X.tobytes()  # or .tostring() if numpy < 1.9
        str_id = '{:08}'.format(2 * k + 1)
        # The encode is only essential in Python 3
        txn_label1.put(str_id.encode('ascii'), datum.SerializeToString())

        X = np.transpose(label_result2, (2, 0, 1))
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[0]
        datum.height = X.shape[1]
        datum.width = X.shape[2]
        datum.data = X.tobytes()  # or .tostring() if numpy < 1.9
        str_id = '{:08}'.format(2 * k + 1)
        # The encode is only essential in Python 3
        txn_label2.put(str_id.encode('ascii'), datum.SerializeToString())

        if k % 10 == 0:
            txn_image.commit()
            txn_label1.commit()
            txn_label2.commit()

            txn_image = env_image.begin(write=True)
            txn_label1 = env_label1.begin(write=True)
            txn_label2 = env_label2.begin(write=True)

    txn_image.commit()
    txn_label1.commit()
    txn_label2.commit()

    # 结束后记住释放资源，否则下次用的时候打不开。。。
    env_image.close()
    env_label1.close()
    env_label2.close()


if __name__ == "__main__":
    temp = cv2.imread(root + "/data/3templates/2.png")
    make_lmdb(temp)

# # 神经网络预测输出
# class FCN(object):
#     def __init__(self, deploy, model):  # 初始化网络,获取网络参数备用
#         caffe.set_mode_gpu()  # caffe.set_mode_cpu()
#         # deploy = "/media/yong/data/cellphone_inspention/cellphone/caffe/deploy_fcn.prototxt"  # deploy文件的路径
#         # model = "/media/yong/data/cellphone_inspention/cellphone/caffe/model/luoding1.caffemodel"  # caffe_model的路径
#         self.net = caffe.Net(deploy, model, caffe.TEST)  # 加载model和network
#         self.net.blobs['data'].reshape(1, 6, 850, 1680)
#
#         # template_path = "/media/yong/data/cellphone_inspention/cellphone/python/templates/"
#         # template = cv2.imread(template_path + r"/2.jpg")
#         # self.template = template[350:1200, 300:1980, :]  # 截取ROI
#         # self.ref = np.transpose(self.template, (2, 0, 1))
#
#     def processing(self, x):  # 处理图像得到mask
#         # image = np.transpose(img, (2, 0, 1))  # 将H*W*C转化为C*H*W
#         # X = np.concatenate((image, self.ref), axis=0)[np.newaxis, :].astype(np.float32)  # 同时将C*H*W转为1*C*H*W
#
#         # im = caffe.io.load_image("xxx.png")  # 加载图片
#         # net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面的预处理操作，并将图片载入到blob中
#         self.net.blobs['data'].data[...] = x  # 执行上面的预处理操作，并将图片载入到blob中
#
#         # 运行S3 网络前向计算
#         self.net.forward()
#         conv5 = self.net.blobs['conv5'].data[0]  # 取出conv5层
#         # result = np.argmax(conv5, axis=0).astype(np.uint8)
#         #
#         # # 运行S4 后处理
#         # # cv2.imwrite("before.jpg", result.astype(np.uint8) * 255)
#         # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         # result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, kernel)
#         # # cv2.imwrite("After.jpg", result.astype(np.uint8) * 255)
#         # # 显示结果
#         # result = cv2.resize(result, (img.shape[1] - 47, img.shape[0] - 47), interpolation=cv2.INTER_NEAREST)
#         # mask = np.zeros((img.shape[0], img.shape[1]))
#         # mask[24:-23, 24:-23] = result
#         return conv5[0, :]
#
#
# # ---图像矫正函数---------------------------------------------------------------------------------------------------------
# # 1. 对定位ROI区域实行互相关操作，完成对齐角点的检测
# # 2. 利用图像矫正手段，对手机ROI区域的图像进行变换
# def image_correct(template, image):
#     L = 201  # 角落的边框尺寸大小
#     Location_ROI = [[(70, 50), (70 + L, 50 + L)],
#                     [(70, 600), (70 + L, 600 + L)],
#                     [(1400, 50), (1400 + L, 50 + L)],
#                     [(1400, 600), (1400 + L, 600 + L)]]
#     standard_points = np.mean(np.array(Location_ROI), axis=1)  # [[ 170,150], [170,700], [1500,150], [1500,700]]
#     transform_points_delta = [(0, 0), (0, 0), (0, 0), (0, 0)]
#     for i in range(4):  # 计算四个角落ROI的位移偏置值
#         # cv2.rectangle(img, Location_ROI[i][0], Location_ROI[i][1], (255, 255, 255))
#         # cv2.rectangle(template, Location_ROI[i][0], Location_ROI[i][1], (255, 255, 255))
#         img_roi = image[Location_ROI[i][0][1]:Location_ROI[i][1][1], Location_ROI[i][0][0]:Location_ROI[i][1][0]]
#         template_roi = template[Location_ROI[i][0][1]:Location_ROI[i][1][1],
#                        Location_ROI[i][0][0]:Location_ROI[i][1][0]]
#         # 3个通道的互相关系数求和
#         similarity = np.abs(
#             np.fft.fftshift(
#                 np.fft.ifft2(np.fft.fft2(img_roi[:, :, 0]) * np.fft.fft2(template_roi[:, :, 0]).conjugate())))
#         +np.abs(
#             np.fft.fftshift(
#                 np.fft.ifft2(np.fft.fft2(img_roi[:, :, 1]) * np.fft.fft2(template_roi[:, :, 1]).conjugate())))
#         +np.abs(
#             np.fft.fftshift(
#                 np.fft.ifft2(np.fft.fft2(img_roi[:, :, 2]) * np.fft.fft2(template_roi[:, :, 2]).conjugate())))
#         # 互相关系数最大的位置，反映出对应ROI的相对位移
#         transform_points_delta[i] = ((np.argmax(similarity) % len(similarity[0])) - (L - 1) / 2,
#                                      np.floor(np.argmax(similarity) / len(similarity[0])) - (L - 1) / 2)
#
#     # print(transform_points_delta)  # 调试打印偏差
#     if np.max(transform_points_delta[:]) > 10:  # 与模板的相对偏移太大，发出警报
#         print("Warning: max displacement >10 pixel, the phone may be placed in a wrong way!!")
#     transform_points = np.array(transform_points_delta) + standard_points + np.random.uniform(-1.5, 1.5, (4, 2))
#     # 计算透视投影变换矩阵,并做矫正
#     M = cv2.getPerspectiveTransform(np.float32(transform_points), np.float32(standard_points))
#     img_correct = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
#     return img_correct, M
