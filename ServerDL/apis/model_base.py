# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import os, re
from ServerDL.cfgs.configfiles import *

digit = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
class Model_base:
    '''模型基类
    所有模型都有四步操作：初始化(init),预处理(preprocess),预测(predict),后处理(postprocess)

    其中predict所有模型通用，其他函数则各模型不同'''
    def __init__(self, key):
        self._key = key
        self._modelcfg = model_register_table[self._key]
        self._device = "cuda"
        self._cpu_device = torch.device("cpu")
        # self.__model = None

    def predict(self, im, **kargs):
        result0, infos = self.preprocess(im, **kargs)
        with torch.no_grad():
            result1 = self._model(result0)
        result2 = self.postprocess(result1, infos, **kargs)
        return result2

    def preprocess(self, im, **kargs):
        return im, None

    def postprocess(self, result, infos, **kargs):
        return result

    def parse_args(self, cfgstr):
        bools = ['True', 'False', 'true', 'false']
        cfgstr = cfgstr
        if os.path.isfile(cfgstr):
            cfgstr = open(cfgstr).read()
        items = cfgstr.split('\n')
        options = {}
        for item in items:
            if '=' not in item or item.strip().startswith('#'):
                continue
            key, val = item.replace(' ', '').split('#')[0].split('=')

            if ',' in val:
                val = val.split(',')
                if digit.match(val[0]):
                    options[key] = list(map(lambda x: int(x) if str.isnumeric(x) else float(x), val))
                else:
                    options[key] = val
            elif str.isnumeric(val):
                options[key] = int(val)
            elif digit.match(val):
                options[key] = float(val)
            elif val in bools:
                options[key] = str.lower(val) == 'true'
            else:
                options[key] = val
        return options