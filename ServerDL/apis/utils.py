import numpy as np
import torch
from ServerDL.cfgs.Protocol import *


def packup(prediction, segms=None, result_type="detection"):
    if result_type == "detection":          # detection
        response_boxes = prediction.bbox.cpu().int().numpy().flatten().tolist()
        response_classes = prediction.get_field("labels").cpu().numpy().flatten().tolist() if prediction.has_field("labels") else None
        response_keyps = prediction.get_field("keypoint").cpu().numpy().flatten().tolist() if prediction.has_field("keypoint") else None  # ???不确定fieldname是不是keypoint
        response_conf = prediction.get_field("scores").cpu().numpy().flatten().tolist() if prediction.has_field("scores") else None

        package = {
            KEY.REQUEST_TYPE: VALUE.REQUEST_TYPE.MODEL_PREDICT,
            KEY.RESPONSE_STATUS: VALUE.RESPONSE_STATUS.SUCCEED,
            KEY.PREDICTION.BOXES: response_boxes,
            KEY.PREDICTION.BOXMODE: prediction.mode,
        }
        if response_classes:
            package[KEY.PREDICTION.CLASSES] = response_classes
        if response_conf:
            package[KEY.PREDICTION.CONFIDENCES] = response_conf
        if response_keyps:
            package[KEY.PREDICTION.KEYPSINFO] = response_keyps,

        if isinstance(segms, torch.Tensor):
            segms = segms.cpu().numpy().astype(np.uint8)
            package[KEY.PREDICTION.MASKINFO] = {KEY.PREDICTION.ROWS: segms.shape[0],
                                        KEY.PREDICTION.COLS: segms.shape[1],
                                      KEY.PREDICTION.CHANNELS: segms.shape[2] if len(segms.shape) == 3 else 1}
    elif result_type == "classification":       # classification
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy().tolist()
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        package = {
            KEY.REQUEST_TYPE: VALUE.REQUEST_TYPE.MODEL_PREDICT,
            KEY.RESPONSE_STATUS: VALUE.RESPONSE_STATUS.SUCCEED,
            KEY.PREDICTION.CLASSES: list(prediction) if isinstance(prediction, list) else [prediction]
        }
    elif result_type == "segmentation":
        assert isinstance(prediction, np.ndarray), "segmentation is no a ndarray"
        segms = prediction.copy()
        package = {
            KEY.REQUEST_TYPE: VALUE.REQUEST_TYPE.MODEL_PREDICT,
            KEY.RESPONSE_STATUS: VALUE.RESPONSE_STATUS.SUCCEED,
            KEY.PREDICTION.MASKINFO: {KEY.PREDICTION.ROWS: segms.shape[0],
                                      KEY.PREDICTION.COLS: segms.shape[1],
                                      KEY.PREDICTION.CHANNELS: segms.shape[2] if len(segms.shape) == 3 else 1}
        }


    print(package)
    return package, segms


def select_top_predictions(predictions, confidence_threshold):
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]



