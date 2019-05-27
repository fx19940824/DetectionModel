###文件目录结构
    ├── api  
    │   ├── DetectronFactory.py: 负责模型发放     
    │   ├── ExceptionHandler.py: 异常捕获及处理  
    │   ├── model_base.py： 模型基类  
    │   ├── model_impl.py： 模型不同类型的具体实现  
    │   ├── transformers.py： 预处理方法（具体模型需要在配置文件中注册transformer）    
    │   ├── postprocess.py： 后处理方法（具体模型需要在配置文件中注册postprocess）   
    │   └── utils.py： 后处理函数和额外的前处理函数等,对结果进行解析和封装返回   
    ├── cfgs  
    │   ├── configfiles.py: 具体模型注册表,部署时只需关注该文件即可  
    │   ├── detectors.py: 模型类别注册表,在model_impl.py中实现的新种类模型需要在此注册
    │   ├── Protocol.py： 传输协议,保证与前段键值一样  
    ├── maskrcnn_benchmark:  maskrcnn源码
    │   └── ...  
    ├── yolo3: yolo源码  
    │   └── ...  
    ├── ganomaly: ganomaly源码  
    │   └── ...  
    ├── weights: 具体模型权重存放路径  
    │   ├── fll_final.pth  
    │   └── yolo_fll_10000.weights  
    ├── test  
    ├── readme.md  
    └── Server.py: 服务器启动入口,通过ide或终端开启服务  



### configfiles.py 说明
    需要 import transformers.py utils.py, 因为要给模型注册处理函数
    注册表：model_register_table
    具体模型名称: {
        model_cfg_string: 模型网络结构,对应的cfg/yaml文件字符串(非目录)
        model_weight_path: 模型权重文件加载路径, yolo必须填写, rcnn在 model_cfg_string 有默认路径
        model_type: 模型类别,目前支持[YOLO,RCNN]
        model_transformer: 前处理函数,根据任务需求可自己在transformers.py中实现并注册于此
        model_handle_function: 后处理函数,同上
        model_network: 分类模型需要的参数, 指定要加载的网络
        }
    
### models.py 说明
    MODELS = {
        "RCNN": {
            "model": MODEL_RCNN,
            "default_params": ["confidence_threshold", "mask_threshold"]
        },
        "YOLO": {
            "model": MODEL_YOLO,
            "default_params": ["confidence_threshold", "nms_threshold"]
        },
        "CLS": {
            "model": MODEL_CLS,
            "default_params": [],
        },
    }
    
    模型类别名称("RCNN"/"YOLO"/"CLS")：对应configfiles.py中的model_type
    model字段: 对应model_impl中的模型类
    default_params字段: 模型初始化可以传入的参数名称
    
### 传输协议见protocol.py
    传输中常见字段：
    model_name：模型名称(不定值)
    request_type：访问类型(0为初始化,1为预测,2为关闭预测模型并退出线程)
    predict_method：预测的方式(0为共享内存, 1为socket, 2为http)
    status：服务器响应动作(0为成功, 1为失败, 2为关闭预测模型并退出线程)
    errinfo: 服务器响应失败时的错误信息(不定值)
    
    预测常见字段:
        访问信息:
            address:共享内存方式中的内存地址
            scale:图片输入大小(部分模型支持多尺度输入,可在预测阶段不更改模型的情况下改变输入图像尺寸)
            cols & rows & channels:
            keypoints: 是否要返回关键点预测结果(0为False,1为True)
            mask: 是否要返回掩码预测结果(0为False,1为True)
        返回信息:
            classes: 预测结果的类别(list)
            confidences: 预测结果的置信度(list)
            boxes: 预测结果的保卫框(list)
            boxmode: 包围框格式(xyxy,xywh)
            keypinfo: 关键点预测结果
            maskinfo: 掩码预测结果
    
    
    