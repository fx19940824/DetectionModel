import glob
import os
import cv2
import numpy as np
from darknet.tools.predictor import YOLODemo




def whole2seg(img, results):
    list = []
    for x1, y1, x2, y2 in results:
        seg = img[y1:y2, x1:x2, :]
        # cv2.imshow("a", seg)
        # cv2.waitKey(0)
        list.append(seg)
    return list


if __name__ == '__main__':
    source_dir = "/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_board"
    target_dir = "/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_whole"
    yolo_cfg = "/home/cobot/caid2.0/python/Main/TrainerDL/Projects/FLL_flaw_detection/cfgs/yolo/n_yolo_fll.cfg"
    yolo_weight = "/home/cobot/caid2.0/python/Main/TrainerDL/Projects/FLL_flaw_detection/weights/n_yolo_fll_5000.weights"

    source_imgs = glob.glob(os.path.join(source_dir, "*/*.png"))
    spliter = YOLODemo(yolo_cfg, yolo_weight, show_img=False)
    for img_path in source_imgs:
        directory = '/'.join(img_path.split('/')[:-1]).replace(source_dir, target_dir)
        if not os.path.exists(directory):
            os.mkdir(directory)
        img = cv2.imread(img_path)
        results = spliter.predict(img).bbox.data.cpu().numpy()
        print(img_path, results.shape[0])
        results = results.astype(np.int)
        seg_list = whole2seg(img, results)
        for idx, seg in enumerate(seg_list):
            path = os.path.join('/'.join(img_path.split('/')[:-1]),
                                img_path.split('/')[-1][:-4]+'_{}.png'.format(idx)).replace(source_dir, target_dir)
            cv2.imwrite(path, seg)
            pass

