
import sys
sys.path.append('/home/cobot/code/caid2.0/python/Algorithm/maskrcnn-benchmark-stable/')

from apis import *
from Utils.parsers import parse_cfg
import argparse


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--configfile",
        help="path to config file",
        default='/home/cobot/caid2.0/python/Main/TrainerDL/cfgs/rcnn/train_rcnn.cfg',
        type=str,
    )

    args = parser.parse_args()
    cfg = parse_cfg(args.configfile)
    if cfg["model_type"] == "yolo":
        pass # not implemented
    elif cfg["model_type"] == "rcnn":
        test_rcnn(cfg)
    elif cfg["model_type"] == "classification":
        pass # not implemented


if __name__ == '__main__':
    main()
