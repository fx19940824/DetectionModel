from maskrcnn_benchmark.tools.predictor import RCNNDemo
from apis.rcnn import cfg_update_and_freeze
from maskrcnn_benchmark.maskrcnn_benchmark.config import cfg
from Utils.parsers import parse_cfg
import cv2

# rcnn 单个图片测试用例

if __name__=='__main__':
    cfgs = parse_cfg("/home/cobot/caid2.0/python/Main/TrainerDL/cfgs/rcnn/train_rcnn.cfg")
    cfg_update_and_freeze(cfgs)
    PD = RCNNDemo(cfg)
    res = PD.run_on_opencv_image(cv2.imread("/media/cobot/00006784000048233/Dataset/fll-seg/train/00000006.jpg"))
    cv2.imshow("a", res)
    cv2.waitKey(0)