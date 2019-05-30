import sys

sys.path.append('/home/fangxin/DetectionModel/Algorithm/maskrcnn_benchmark_stable/')

from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor import MaskRCNNDemo
import glob
import os
from tqdm import tqdm
import time
import torch

def completeDir(dir):
    if dir[-1]!='/':
        dir += '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

if __name__ == '__main__':
    config_file = "/home/fangxin/DetectionModel/TrainerDL/Projects/BuildingDetection/n_rcnn.yaml"
    dir_path_positive = "/home/fangxin/Dataset/building/test/"
    output_path_positive = "/home/fangxin/DetectionModel/TrainerDL/Projects/BuildingDetection/result/"

    output_path_positive = completeDir(output_path_positive)

    isWrite = True
    postprocess = False
    calculateTime = False
    maskiou_on = False

    cfg.merge_from_file(config_file)

    confidence=[0.7]

    print(config_file)
    # torch.cuda.set_device(0)
    demo = MaskRCNNDemo(cfg)
    list_imgpath = glob.glob(dir_path_positive + "*.png")
    print("test on {} positive".format(len(list_imgpath)))
    result=np.zeros((len(confidence)))
    avg_write_time=0
    max_write_time=0
    for image_path in tqdm(list_imgpath):

        pil_image = Image.open(image_path).convert('RGB')
        image = np.array(pil_image)
        segms, predictions, isDetected = demo.run_on_opencv_image(image, confidence, isWrite, postprocess, calculateTime,maskiou_on)
        result+=np.array(isDetected)
        if isWrite:
            for prediction,detect in zip(predictions,isDetected):
                if detect==1:
                    img = Image.fromarray(prediction)
                    img.save(output_path_positive + os.path.split(image_path)[1])

            if calculateTime:
                t0=time.time()

            #if segms.sum() != 0:
            #    img = Image.fromarray(segms.cpu().numpy())
            #    img.save(output_path_positive + os.path.split(image_path)[1])

            if calculateTime:
                t1=time.time()
                print("write time：{}".format((t1 - t0)*1000))
                avg_write_time+=(t1-t0)
                max_write_time=max(max_write_time,t1-t0)

    if calculateTime:
        print('avg detection time:{}'.format(demo.avgtime[0]*1000/len(list_imgpath)))
        print('avg postprocess time:{}'.format(demo.avgtime[1]*1000/len(list_imgpath)))
        print('avg prepare write time:{}'.format(demo.avgtime[2]*1000 / len(list_imgpath)))
        print('avg write time:{}'.format(avg_write_time*1000 / len(list_imgpath)))

        print('max detection time:{}'.format(demo.maxtime[0] * 1000))
        print('max postprocess time:{}'.format(demo.maxtime[1] * 1000))
        print('max prepare write time:{}'.format(demo.maxtime[2] * 1000))
        print('max write time:{}'.format(max_write_time * 1000))

    print("dataset positive")
    for i in range(len(confidence)):
        notdetect=len(list_imgpath)-result[i]
        print("confidence {}: detect {}, not detect {}".format(confidence[i],result[i],notdetect))
