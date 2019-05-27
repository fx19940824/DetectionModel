from ServerDL.apis.transformers import *
from ServerDL.apis.postprocesses import *
from Algorithm.classifications.utils.model_factory import classification
from Algorithm.segmentation.deeplabv3plus.modeling.deeplab import DeepLab


model_register_table = dict()

model_register_table["maskrcnn_box"] = {
        "model_weight_path": "/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Projects/bag/resnext101/model_0660000.pth",  #  选填，会覆盖model_cfg_string中WEIGHT的路径
        "model_cfg_file":"/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Projects/bag/resnext101/n_rcnn.yaml",
        "model_type": "RCNN",
        "model_transformer": build_transform_maskrcnn,
        "model_handle_function": build_postprocess_plgdetection,
        "model_network": "",
        "confidence_threshold": 0.9
}

model_register_table["maskrcnn_bag"] = {
        "model_weight_path": "/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Projects/bag/resnext101/model_0660000.pth",  #  选填，会覆盖model_cfg_string中WEIGHT的路径
        "model_cfg_file":"/media/cobot/30b0f4a0-3376-4f8f-b458-9c6857504361/Projects/bag/resnext101/n_rcnn.yaml",
        "model_type": "RCNN",
        "model_transformer": build_transform_maskrcnn,
        "model_handle_function": build_postprocess_plgdetection,
        "model_network": "",
        "confidence_threshold": 0.7
}

model_register_table["maskrcnn_tube"] = {
        "model_weight_path": "/home/cobot/tube_model/model_0505000.pth",
        "model_cfg_file": "/home/cobot/tube_model/n_rcnn.yaml",
        "model_type": "RCNN",
        "model_transformer": build_transform_maskrcnn,
        "model_handle_function": build_postprocess_plgdetection,
        "model_network": "",
        "confidence_threshold": 0.8
}

model_register_table["CLS"] = {
        "model_cfg_string":
            '''
                modelname = resnet18
                classes = 2
                img_size = 224
                lr = 0.001
                batchsize = 32
                epochs = 10
                freeze_layers = 0
                is_Train = True
                transformer = default
                half = True
            ''',
        "model_weight_path": "/home/cobot/caid2.0/python/Main/ServerDL/weights/test.pt",
        "model_type": "CLS",
        "model_transformer": build_transform_cls,
        "model_handle_function": build_postprocess_cls,
        "model_network": classification
    }

model_register_table["DeepLabv3+"] = {
        "model_cfg_string":
            '''
                num_classes = 2
                backbone = mobilenet
                output_stride = 16
                sync_bn = False
                freeze_bn = False
                img_size = 257
            ''',
        "model_weight_path": "/home/cobot/model_best.pth.tar",
        "model_type": "SEG",
        "model_transformer": build_transform_seg,
        "model_handle_function": build_postprocess_seg,
        "model_network": DeepLab
}



