
import sys
sys.path.append('/home/fangxin/DetectionModel/')


from apis import *
from Utils.parsers import parse_cfg
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--configfile",
        help="path to config file",
        default='/home/fangxin/DetectionModel/TrainerDL/Projects/BuildingDetection/train_rcnn.cfg',
        type=str,
    )

    args = parser.parse_args()
    cfgs = parse_cfg(args.configfile)

    torch.cuda.set_device(cfgs["gpus"])

    if cfgs["model_type"] == "yolo":
        train_darknet(cfgs)
    elif cfgs["model_type"] == "rcnn":
        train_rcnn(cfgs)
    elif cfgs["model_type"] == "classification":
        train_cls(cfgs)



if __name__ == '__main__':
    main()
