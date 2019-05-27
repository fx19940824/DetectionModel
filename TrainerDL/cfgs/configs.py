from maskrcnn_benchmark.maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
import os
registry = {
    "yolo_test": {  # model name
        "model_type": "yolo",
        "cfgpath": "/home/cobot/caid2.0/python/TrainerDL/cfgs/yolo_fll.cfg",     # yolo网络结构路径
        "train_dir": "/home/cobot/Dataset/fll-whole/FLL_416",                               # 数据集路径，自动生成数据列表并动注册到data文件
        "val_dir": "/home/cobot/Dataset/fll-whole/FLL-VAL",
        "names": ["good"],                                                                  # 自动生成类别文件并注册到data文件
        "init_weights": "/home/cobot/caid2.0/python/TrainerDL/darknet/weights/yolo_fll.backup",                             # 初始化权重路径
        "backup": "/home/cobot/caid2.0/python/TrainerDL/darknet/weights",                   # 训练中间与最终权重保存路径
        "classes": 1,                                                                       # 类别数，需要注册到data文件中
        "gpus": [0]
    },

    "rcnn_test": {  # model name
        "model_type": "rcnn",                                                               # 模型类型（目前支持rcnn和yolo）
        "cfgpath": "/home/cobot/caid2.0/python/TrainerDL/cfgs/FLLMODELTEST.yaml",           # 指定yaml路径
        "train_img_dir": "/home/cobot/Dataset/fll-seg/train",                               # 表明路径后将自动注册到paths_catalog.py中
        "train_ann_dir": "/home/cobot/Dataset/fll-seg/annotations/instances_fll_train.json",
        "val_img_dir": "/home/cobot/Dataset/fll-seg/val",
        "val_ann_dir": "/home/cobot/Dataset/fll-seg/annotations/instances_fll_val.json",
        "gpus": [0],    # load_rank

        "opts": ["PATHS_CATALOG",                                                           # yaml文件中修改的参数，PATHS_CATALOG路径
                 os.path.join(os.path.dirname(__file__), "paths_catalog.py"),               # 必须修改，因为需要自动注册训练集，这里基本不需要改动
                 "OUTPUT_DIR",                                                              # 结果输出的路径
                 "/home/cobot/caid2.0/python/TrainerDL/weights"
                 ]
    },

    "rcnn_test2": {  # model name
        "model_type": "rcnn",                                                               # 模型类型（目前支持rcnn和yolo）
        "cfgpath": "/home/cobot/caid2.0/python/TrainerDL/cfgs/FLLMODELTEST.yaml",           # 指定yaml路径
        "train_img_dir": "/home/cobot/Dataset/fll-test/train",                               # 表明路径后将自动注册到paths_catalog.py中
        "val_img_dir": "/home/cobot/Dataset/fll-test/val",
        "gpus": [0],    # load_rank
        "ratio": 0.3,

        "opts": ["PATHS_CATALOG",                                                           # yaml文件中修改的参数，PATHS_CATALOG路径
                 os.path.join(os.path.dirname(__file__), "paths_catalog.py"),               # 必须修改，因为需要自动注册训练集，这里基本不需要改动
                 "OUTPUT_DIR",                                                              # 结果输出的路径
                 "/home/cobot/caid2.0/python/TrainerDL/weights"
                 ]
    },

    "ganomaly_test": {
        "model_type": "ganomaly",
        "cfgpath": "/home/cobot/caid2.0/python/TrainerDL/cfgs/ganomaly_test.cfg",
        "train_img_dir": "/home/cobot/Dataset/fll-black/train",
        "val_img_dir": "/home/cobot/Dataset/fll-black/test",
        "gpus": [0],
        "ouput_dir": "/home/cobot/test"
    }
}

