#配置文件说明
maskrcnn的拓展功能都可以通过配置文件以开关的方式启动,主要有如下几个部分

## 损失权重的配置
模型中所有的损失都需要在`_C.MODEL.LOSS_WEIGHTS`中标出，该项会在训练中统一为每个损失乘上比例系数。
若在后续开发中需要添加新的损失函数，也需要在此处标出,并在`train_net.py`中添加对应关系
```
_C.MODEL.LOSS_WEIGHTS = CN()
_C.MODEL.LOSS_WEIGHTS.LOSS_OBJECTNESS = 1
_C.MODEL.LOSS_WEIGHTS.LOSS_RPN_BOX_REG = 1
_C.MODEL.LOSS_WEIGHTS.LOSS_CLASSIFIER = 1
_C.MODEL.LOSS_WEIGHTS.LOSS_BOX_REG = 1
_C.MODEL.LOSS_WEIGHTS.LOSS_MASK = 1
_C.MODEL.LOSS_WEIGHTS.LOSS_MASKIOU = 1
_C.MODEL.LOSS_WEIGHTS.LOSS_KEYPOINTS = 1
```

## non-local的配置
non-local只在resnet中增加，
```
_C.MODEL.BACKBONE.NL_TYPE='none'
```

## 形变卷积的配置
由于官方已经更新形变卷积功能，所以采用官方的实现方法。不同的基础网络需要专门添加形变卷积功能，
官方实现中也是在RESNETS网络中添加了该可选项，相关参数如下
```
_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1
```