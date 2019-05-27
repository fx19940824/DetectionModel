#!/usr/bin/python2
# coding:utf-8

import numpy as np
import cv2
import os
import shutil
import sys
from data_set_maker_screw import make_lmdb

# sys.path.append(r'/home/cobot/cellphone_project/caffe/caffe-master/python')
# sys.path.append(r'/home/yong/caffe-master/python')
sys.path.append(r'/home/cobot/cellphone_project/caffe/caffe-master/python')
# "/home/cobot/cellphone_project/caffe/caffe-master/python"
import caffe


def solver_maker(solver_file, net_file, maxT, prefix):
    sovler_string = {}  # sovler存储
    # solver_file = my_project_root + 'solver.prototxt'                        #sovler文件保存位置
    sovler_string['net'] = '\"' + net_file + '\"'  # train.prototxt位置指定
    # sovler_string.test_net.append(my_project_root + 'test.prototxt')         #test.prototxt位置指定
    sovler_string['test_iter'] = '1'  # 测试迭代次数
    sovler_string['test_interval'] = str(99999999)  # 每训练迭代test_interval次进行一次测试
    sovler_string['base_lr'] = str(0.005)  # 基础学习率 0.005
    sovler_string['momentum'] = str(0.99)  # 动量 0.99
    sovler_string['weight_decay'] = str(0.000002)  # 权重衰减
    sovler_string['lr_policy'] = '"step"'  # 学习策略
    sovler_string['stepsize'] = str(2000)  # 2000
    sovler_string['gamma'] = str(0.8)  # 的
    sovler_string['display'] = str(10)  # 每迭代display次显示结果
    sovler_string['max_iter'] = str(maxT)  # 最大迭代数
    sovler_string['snapshot'] = str(np.minimum(np.int(maxT), 1000))  # 保存临时模型的迭代数
    # sovler_string.snapshot_format = 0                                        #临时模型的保存格式,0代表HDF5,1代表BINARYPROTO
    sovler_string['snapshot_prefix'] = '\"' + prefix + '\"'  # 模型前缀
    sovler_string['solver_mode'] = str('GPU')  # 优化模式

    with open(solver_file, 'w') as f:
        for key, value in sorted(sovler_string.items()):
            if not (type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))


def write_sh(sh_file, solver_file, snapshot):
    with open(sh_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`.log\n")
        f.write("CAFFE=/home/cobot/cellphone_project/caffe/caffe-master/build/tools/caffe\n")
        f.write("$CAFFE train --solver=%s  %s | tee $LOG\n" % (solver_file, snapshot))


caffe.set_device(0)
caffe.set_mode_gpu()

iterStep = 2000
root = "/home/cobot/cellphone_project"

for iter in range(5, 20):
    # # 生成数据
    if iter < 1:
        temp = cv2.imread(root + "/data/3templates/2.png")
        make_lmdb(temp)

    # ---------------------------网络1-----------------------------------------------------------------------
    # 生成solver file
    solver_file = root + "/caffe/screw1/screw_fcn_solver1.prototxt"
    net_file = root + "/caffe/screw1/screw_fcn1.prototxt"
    maxT = (iter + 1) * iterStep
    prefix = root + "/caffe/screw1/model/ScrewNet1"
    solver_maker(solver_file, net_file, maxT, prefix)

    # 生成sh训练脚本
    sh_file = root + "/caffe/screw1/train_screw_fcn1.sh"
    if iter == 0:
        snapshot = ""
        # snapshot="--weights="+root+"/caffe/screw1/model/ScrewNet1_iter_"+str(iter*iterStep)+".caffemodel"
    else:
        snapshot = "--snapshot=" + os.path.dirname(root) + "/caffe/screw1/model/ScrewNet1_iter_" + str(
            iter * iterStep) + ".solverstate"
    write_sh(sh_file, solver_file, snapshot)
    # # os.system('sh ' + sh_file)

    # 使用pycaffe训练
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_file)
    if iter > 0:
        solver_state = root + "/caffe/screw1/model/ScrewNet1_iter_" + str(iter * iterStep) + ".solverstate"
        solver.restore(solver_state)
    solver.solve()
    del solver  # 删除solver 释放显存
    # np.delete(solver)
    # 拷贝结果
    shutil.copyfile(root + "/caffe/screw1/model/ScrewNet1_iter_" + str((iter + 1) * iterStep) + ".caffemodel",
                    root + "/caffe/screw1/model/ScrewNet1" + ".caffemodel")

    # ---------------------------网络2-----------------------------------------------------------------------
    # 生成solver file
    solver_file = root + "/caffe/screw2/screw_fcn_solver2.prototxt"
    net_file = root + "/caffe/screw2/screw_fcn2.prototxt"
    maxT = (iter + 1) * iterStep
    prefix = root + '/caffe/screw2/model/ScrewNet2'
    solver_maker(solver_file, net_file, maxT, prefix)

    # # 使用sh脚本训练Caffe网络
    # sh_file = root + "/caffe/screw2/train_screw_fcn2.sh"
    # if iter == 0:
    #     snapshot = ""
    # else:
    #     snapshot = "--snapshot=" + root + "/caffe/screw2/model/ScrewNet2_iter_" + str(iter * iterStep) + ".solverstate"
    # write_sh(sh_file, solver_file, snapshot)
    # os.chdir(root + "/caffe/screw2")
    # # os.system('sh ' + sh_file)

    # 使用pycaffe训练Caffe网络
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_file)
    if iter > 0:
        solver_state = root + "/caffe/screw2/model/ScrewNet2_iter_" + str(iter * iterStep) + ".solverstate"
        solver.restore(solver_state)
    solver.solve()
    del solver  # 删除solver 释放显存

    # 拷贝结果
    shutil.copyfile(root + "/caffe/screw2/model/ScrewNet2_iter_" + str((iter + 1) * iterStep) + ".caffemodel",
                    root + "/caffe/screw2/model/ScrewNet2" + ".caffemodel")


