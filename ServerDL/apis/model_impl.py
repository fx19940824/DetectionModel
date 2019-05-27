# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
from argparse import Namespace
from ServerDL.apis.model_base import Model_base
from ServerDL.apis.utils import packup, select_top_predictions
import time
import sys
sys.path.insert(0, '/home/cobot/code/caid2.0/python/Algorithm/maskrcnn-benchmark-stable')

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

class MODEL_RCNN(Model_base):
    def __init__(self, key, **kwargs):
        Model_base.__init__(self, key)

        confidence_threshold = self._modelcfg.get("confidence_threshold", 0.7)
        weights_path = self._modelcfg.get("model_weight_path", "")
        self._cfg = cfg.clone()
        self._cfg.merge_from_file(self._modelcfg["model_cfg_file"])

        if weights_path:
            self._cfg.merge_from_list(["MODEL.WEIGHT", weights_path])

        self._model = build_detection_model(self._cfg)
        self._model.eval()
        self._model.to(self._device)

        checkpointer = DetectronCheckpointer(self._cfg, self._model)
        _ = checkpointer.load(self._cfg.MODEL.WEIGHT)

        self._confidence_threshold = confidence_threshold
        self.build_transform = self._modelcfg["model_transformer"]
        self._transformer = self.build_transform(self, self._cfg.INPUT.MIN_SIZE_TEST)
        self._postprocess = self._modelcfg["model_handle_function"](self)
        self._masker = Masker()

    def preprocess(self, im, **kargs):
        '''maskrcnn预处理
        将单通道图片转换成3通道，然后进行数据增强'''
        im = im.squeeze()
        if len(im.shape) == 2:
            im = np.concatenate([im[...,np.newaxis], im[...,np.newaxis], im[...,np.newaxis]], axis=2)
        ori_size = im.shape[:-1]
        im = self._transformer(im)
        image_list = to_image_list(im, self._cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self._device)
        return image_list, {"ori_size": ori_size}

    def postprocess(self, result, infos, **kargs):
        result = [o.to(self._cpu_device) for o in result]
        result = select_top_predictions(result[0], self._confidence_threshold)
        prediction, segms = self._postprocess(result, infos["ori_size"], self._masker)
        return packup(prediction, segms)

class MODEL_CLS(Model_base):
    def __init__(self, key, **kwargs):
        Model_base.__init__(self, key)
        self.args = Namespace(**self.parse_args(self._modelcfg["model_cfg_string"]))

        self._model = self._modelcfg["model_network"](self.args)
        weights = torch.load(self._modelcfg["model_weight_path"])

        if weights.get("state_dict", False):
            weights = weights['state_dict']
        self._model.load_state_dict(weights)
        self._model.eval()
        self._model.to(self._device)

        self.build_transform = self._modelcfg["model_transformer"]
        self._transformer = self.build_transform(self)

        self._postprocess = self._modelcfg["model_handle_function"](self)
        pass

    def preprocess(self, im, **kargs):
        im = self._transformer(im)
        im = im.to(self._device)
        return im, None

    def postprocess(self, result, infos, **kargs):
        result = self._postprocess(result)
        result, _ = packup(result, result_type="classification")
        return result, None


class MODEL_SEG(Model_base):
    def __init__(self, key, **kwargs):
        Model_base.__init__(self, key)
        self.args = Namespace(**self.parse_args(self._modelcfg["model_cfg_string"]))

        self._model = self._modelcfg["model_network"](self.args)
        weights = torch.load(self._modelcfg["model_weight_path"])

        if weights.get("state_dict", False):
            weights = weights['state_dict']
        self._model.load_state_dict(weights)
        self._model.eval()
        self._model.to(self._device)

        self.build_transform = self._modelcfg["model_transformer"]
        self._transformer = self.build_transform(self)

        self._postprocess = self._modelcfg["model_handle_function"](self)

    def preprocess(self, im, **kargs):
        ori_shape = im.shape[:2]
        im = self._transformer(im)
        im = im.to(self._device)
        return im, ori_shape

    def postprocess(self, result, infos, **kargs):
        result = self._postprocess(result, infos)
        return packup(result, result_type="segmentation")


if __name__ == '__main__':
    import cv2,  glob
    m = MODEL_RCNN("maskrcnn_tube")
    imgs = glob.glob("/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Dataset/tube-test/*.png")
    for path in imgs:

        print(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        t = time.time()
        res, mask = m.predict(img)
        print("time====================+>{}".format(time.time() - t))
        mask = cv2.resize(mask, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_CUBIC)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for contour in range(len(contours)):
            cv2.drawContours(img, contours, contour, (0, 0, 255), 3)
        cv2.imwrite(path.replace(".png", "_out.png"), img)

        pass
