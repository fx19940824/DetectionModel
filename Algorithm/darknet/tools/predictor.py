from darknet.models import Darknet
from darknet.utils.utils import non_max_suppression
from torchvision import transforms as T
from maskrcnn_benchmark.structures.bounding_box import BoxList
import numpy as np
import torch
import os
import cv2

class YOLODemo:
    def __init__(self, cfg, weight, confidence_threshold=0.5, nms_threshold=0.4, show_img=True):

        self._device = torch.device("cuda:0")
        if os.path.isfile(cfg):
            cfg = open(cfg).read()
        self._model = Darknet(cfg)
        self._model.load_weights(weight)
        self._model.eval()
        self._model.to(self._device)
        self._shape = self._model.img_size
        self.show_img = show_img
        self._transformer = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((self._shape, self._shape)),
            T.ToTensor(),
            T.Lambda(lambda x: x[[2, 0, 1]]),
            T.Lambda(lambda x: x.float().unsqueeze(0))
        ])

        self._nms_threshold = nms_threshold
        self._confidence_threshold = confidence_threshold

    def predict(self, img, save_file=None, printinfo=False):
        ori_img = img.copy()
        result0, ori_shape = self.preprocess(img,)
        with torch.no_grad():
            result1 = self._model(result0)
        result2 = self.postprocess(result1, ori_shape)
        for box, lb, conf in zip(result2.bbox, result2.extra_fields["labels"], result2.extra_fields["scores"]):
            lb = int(lb.numpy())
            conf = float(conf.numpy())
            box = tuple(box.int().numpy().tolist())
            cv2.rectangle(ori_img, box[:2], box[2:],tuple(np.random.randint(0, 255, 3).tolist()), 1)
            cv2.putText(ori_img, "cls:%d, p:%.2f" % (lb, conf), box[:2], cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0))
            if printinfo:
                print("location:  %20s,    label:%3d,    confidnce:  %8f" % (str(box), lb, conf))
        if self.show_img:
            cv2.imshow("result", ori_img)
            cv2.waitKey(0)
        if save_file:
            cv2.imwrite(save_file, ori_img)
        return result2

    def preprocess(self, im, **kargs):
        ori_shape = im.shape[:2]
        im = self._transformer(im)
        im = im.to(self._device)
        return im, ori_shape

    def postprocess(self, result, ori_shape, **kargs):
        shape = result.shape
        result = non_max_suppression(result, num_classes=self._model.num_cls, nms_thres=self._nms_threshold)
        if result:
            result = torch.Tensor([]).reshape([0, 7])
        else:
            result = result[0].cpu()
        result = self.yolo2boxlist(result)
        result = result.resize(ori_shape[::-1])
        print(result.shape[0])
        return result

    def yolo2boxlist(self, bbox):
        fields = bbox[..., 4:]
        bbox = bbox[..., :4]
        boxlist = BoxList(bbox, (self._shape, self._shape), mode='xyxy')
        boxlist.add_field("labels", fields[..., 2].int())
        boxlist.add_field("scores", fields[..., 0])
        return boxlist
