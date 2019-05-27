from Algorithm.darknet.utils import YOLO_Kmeans, parse_from_config
# yolo anchor box 聚类, 需要指定train_cfg文件和 anchor 个数

if __name__=='__main__':
    train_yolo_cfg = "/home/cobot/caid2.0/python/Main/TrainerDL/Projects/FLL_flaw_detection/cfgs/yolo/train_yolo.cfg"
    pfl = parse_from_config(train_yolo_cfg)
    cluster_number = 2
    kmeans = YOLO_Kmeans(cluster_number, pfl)
    kmeans.txt2clusters()
