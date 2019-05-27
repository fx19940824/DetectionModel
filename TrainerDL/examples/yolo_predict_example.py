from Algorithm.darknet.tools.predictor import YOLODemo
import cv2
import glob
import os
if __name__=='__main__':
    md_yolo = YOLODemo("/home/cobot/n_yolo_nissan.cfg",
                       "/home/cobot/n_yolo_nissan_50000.weights",
                       show_img=False)
    imgs = glob.glob("/home/cobot/val/*.png")
    for path in imgs:
        img = cv2.imread(path)

        res = md_yolo.predict(img)
        for id, bbox in enumerate(res.bbox):
            x1, y1, x2, y2 = map(int,  bbox)
            subimg = img[y1:y2, x1:x2, :]
            cv2.imwrite(path.replace('valid_board/', 'valid/').replace('.png', '_%d.png' % id), subimg)
