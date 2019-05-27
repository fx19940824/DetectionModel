##empty_labelme_generate.py
    为没有标注的图像生成空的标注文件，主要作用是将正常样本加入训练
    输入：图片文件夹路径
    输出：对应的json文件
    
##empty_mask_generate.py
    为图像生成空的mask，主要作用是将正常样本加入语义分割的训练
    输入：图片文件夹路径
    输出：对应的json文件
    
##coco2mask.py
    用于将coco数据集转换成某些语义分割模型训练所用的二值mask
    输入：指定coco数据集路径
    输出：在coco数据集下以_mask.png结尾的图片
    
##labelme2coco.py
    用于将labelme标注文件转换成coco数据集,模型训练时需要
    输入：输入输出路径
    
##labelme2mask.py
    用于将labelme标注数据转换成某些语义分割模型训练所用的二值mask
    输入：路径
    输出：以mask.png结尾的图片
    
##mask2labelme.py
    用于将mask文件转换回labelme标注文件，一般用于自动生成mask后，手工检查标注。
    比如用一个初步的模型来生成掩码，并进一步生成标注，可以加快标注速度
    输入：mask图片路径
    输出：json文件
    
##translate_labelme_label.py
    用于转换labelme标注文件的标签
    输入：路径，就地修改
    
##remove_unmarked_img.py
    用于将label标注文件夹中的未标注文件(没有对应的json文件)移到新文件夹
    输入：路径
    输出：新文件夹，包含了未标注的png文件
    
##labelme_split.py
    用于转换labelme标注文件中的标注块截取出来，为下一步合成数据做准备
    输入：路径
    输出：新文件夹，包含了标注块和相应的json文件
    
##clip_voc_data.py
    用于根据voc标注的数据(yolo)，裁剪出相应的图像块
    输入：路径
    输出：原路径下保存图像块