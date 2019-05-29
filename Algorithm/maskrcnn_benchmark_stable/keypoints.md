# 关键点检测使用说明

## 检测原理
关键点检测是一条与掩码预测相似的单独分支，需要确定检测的关键点个数，模型会为每一个关键点生成
一个mask，其上会有单独的一个点当做该关键点的位置
## 数据标注
关键点数据标注依据COCO数据集的标注格式，与实例目标检测不同之处在于：
- annotations字段中增加了keypoints，该字段每三个数字为一组，前两个代表关键点的坐标，第三个
代表关键点的状态（0：没有标注；1：标注不可见；2：可见标注），列表中数字个数应为num_keypoints
中所写关键点个数的三倍
- annotations字段中num_keypoints字段表示关键点的个数

以下是annotations的一个标注实例
```
{
	"segmentation": [[125.12,539.69,140.94,522.43...]],
	"num_keypoints": 10,
	"area": 47803.27955,
	"iscrowd": 0,
	"keypoints": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,142,309,1,177,320,2,191,398...],
	"image_id": 425226,"bbox": [73.35,206.02,300.58,372.5],"category_id": 1,
	"id": 183126
},
```
- categories字段中增加keypoints字段，标识每一个关键点的名称，列表中字符串个数应与关键点个数相同
- categories字段中skeleton字段表示关键点之间的连接性，每两个数字为一组，数值代表关键点的标号

以下是categories的一个标注实例
```
{
	"supercategory": "person",
	"id": 1,
	"name": "person",
	"keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
	"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
}
```

TrainerDL中labelme2coco.py中有根据labelme标注的数据转换成关键点的功能，函数中的参数isKeypoints
会根据cfg（不是yaml）中的isKeypoints的值来选择是否采用关键点转换。转换中只会根据标注
的点生成成annotations中的keypoints和num_keypoints字段，但关键点的类别名会根据标注顺序以
自然数代替，skeleton则会根据标注顺序从第一个点连至最后一个点。最后实际上起作用的只有annotations
中的keypoints和num_keypoints字段。

##使用方法
- 在cfg中将isKeypoints置为True，采用关键点数据标注转换
- 在yaml中将`_C.MODEL.KEYPOINT_ON`置为True,然后在`_C.MODEL.ROI_KEYPOINT_HEAD`中修改关键点
分支中的网络参数，与mask分支类似。需注意NUM_CLASSES应与关键点的个数相同。

keypoints检测代码可拓展性不强，针对不同的目标需要单独编写程序。如果有新的检测目标，需要在
如下部分编写程序：
- 在`./maskrcnn_benchmark/structures/keypoint.py`的`PersonKeypoints`类中需要写明
每一个关键点的名称、镜像关系，
- 在`./maskrcnn_benchmark/structures/keypoint.py`的`kp_connections`函数中，
kp_lines变量需要给出关键点之间的连接关系（实际上就是skeleton）。