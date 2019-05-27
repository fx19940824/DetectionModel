from Algorithm.classifications.utils.model_factory import *
from Algorithm.classifications.utils.transformers import default_transformer
from Algorithm.classifications.utils import transformers as custom_transformer
from ..Utils.parsers import parse_cfg
from argparse import Namespace

def train_cls(cfgs):
    init_weight = cfgs.get("init_weight", None)
    cfgs = Namespace(**cfgs)
    params = parse_cfg(cfgs.cfgpath)
    params = Namespace(**params)
    if params.is_Train:
        if params.transformer == "default":
            transformer = default_transformer(params.img_size)
        else:
            transformer = getattr(custom_transformer, params.transformer)(params.img_size)   #自定义的预处理把transformer文件放到配置文件中！

        print(transformer)
        startTraining(params.modelname, params.classes, params.freeze_layers, transformer, cfgs.img_dir,
                      params.batchsize, cfgs.weight_out, params.epochs, params.lr, weights=init_weight)

