##Install
    pip install -r requirements.txt 依赖包
    python setup.py 完成C++端so库的编译
##该模块包含了yolo,mask-rcnn和分类模块的训练

##trainer.py
    训练入口,通过在终端或ide中执行train(testregistry["模型名称"])来进行训练  
    终端训练: python trainer.py

##cfgs
***yolo：***  

    anchor聚类
    examples/yolo_anchor_example.py

    train_*.cfg文件(训练配置文件)
    
    1. 数据集路径，自动生成数据列表并动注册到data文件  
    *train_dir & val_dir:"/home/cobot/Dataset/fll-whole/FLL_416"    
    2. 自动生成类别文件并注册到data文件  
    *names: car,bike,...
    3. 初始化权重路径  
    *init_weights: "weights/yolov3_tiny_fll_init.weights"   
    4. 训练中间与最终权重保存路径  
    *backup": /home/cobot/caid2.0/python/TrainerDL/darknet/weights"
    5. 类别数，需要注册到data文件中  
    *classes: 1  
    6. 其余参数（超参数）见cfg文件  
    
    网络配置文件：
    n_*.cfg文件（网络结构及超参数文件）
    首先修改batch和subdivisions，建议batch=64, subdivisions=8（1080ti显存最多支持到batch=96 subdivisons=8 random=0）
    random为多尺度训练如图像中的物体大小不一差距很大的话，建议开启。但是比较占用显存
    如果batch太小数据量也很少会导致训练时有大量nan
    再修改3个yolo层前面的卷积层的filter大小，改为 (class数量 + 5) * 3. etc.如分3类 则(3+5)*3 = 24
    再改yolo层中的class数量和anchor box（前述步骤生成）

    单图片可视化测试
    examples/yolo_predict_example.py

***rcnn：***
   
    train_*.cfg文件(训练配置文件)
    1. 数据集路径, 表明路径后将自动注册到paths_catalog.py中  
    *train_img_dir: "/train",   
    train_ann_dir: "instances_fll_train.json",  
    val_img_dir: "/val",  
    val_ann_dir: "instances_fll_val.json",  
    注： 支持三种数据分配方式
    labelme: 填写train_img_dir 与 val_img_dir字段, 图片与标注文件同目录下一一对应
    ————
    coco： 全部字段都需要填写, 需要指出训练集和验证集分别对应的标注文件
    ————
    未划分训练集与验证集的labelme格式： 填写train_img_dir即可，标注文件同labelme
    *ratio: 该类数据集需要在配置中给出 val/dataset 的比例,默认为0.2
    ————
    
    
    2. 其余参数（超参数）见n_*.yaml文件和maskrcnn_benchmark/configs/default.py  
    
    配置文件：
    n_*.yaml文件（网络结构及超参数文件）
    *paths_catalog.py(数据集注册文件)
    
***ganomaly***
    
    1. 模型类别
    model_type: ganomaly
    2. ganomaly没有网络文件,配置文件中的为超参数文件
    cfgpath: /home/cobot/caid2.0/python/TrainerDL/cfgs/n_ganomaly.cfg
    3. 正常数据路径和验证数据路径(正常数据的文件夹必须以0.开头,异常数据的文件夹必须以1.开头)
    train_img_dir: /home/cobot/Dataset/fll-black/train
    val_img_dir: /home/cobot/Dataset/fll-black/test
    4. 
    gpus = 0
    5. 网络权重文件和生成的假照片输出路径
    ouput_dir = /home/cobot/test

    
##darknet  
    yolo 训练接口接口与源码

##maskrcnn_benchmark
    maskrcnn 训练接口与源码

##训练步骤
    1. 准备训练数据和验证数据, 分为两个文件夹存放,标注文件与图片放在同一目录下
    2. 准备网络结构文件(cfg/yaml)和训练配置文件(cfg), 修改某些超参数
    3. 运行trainer.py 并指定训练配置文件

###yolo cfg细则
    batch=64  
    subdivisions=2（根据gpu显存进行更改，显存越大subdivision值越小）  
    width=416     （输入图像尺寸）   
    height=416  
      
    [convolutional]  # [yolo]层的上一个卷积层参数  
    size=1  
    stride=1  
    pad=1  
    filters=12  # filters 个数由[yolo]层决定 计算公式 filters = (4+1+classes)*num  
    activation=linear  
    
    [yolo]
    mask = 0, 1  
    anchors = 65,45,  45,65  # anchor box 的长宽
    classes=1  # 类别数量  
    num=2  # anchor box 的数量  
    jitter=.3  
    ignore_thresh = .7  
    truth_thresh = 1  
    random=0  #  设置为0

### rcnn yaml细则
    待补充
