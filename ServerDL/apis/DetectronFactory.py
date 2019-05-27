# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from ServerDL.cfgs.configfiles import *
from ServerDL.apis.model_impl import *


MODELS = {
    "RCNN": {
        "model": MODEL_RCNN
    },
    "CLS": {
        "model": MODEL_CLS
    },
    "SEG": {
        "model": MODEL_SEG
    },
}

class DetectronFactory:

    @staticmethod
    def get(data):
        predictor = MODELS[model_register_table[data[KEY.MODEL_NAME]]["model_type"]]
        model = predictor["model"]
        return model(data[KEY.MODEL_NAME], **data)
