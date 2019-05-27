

# 键
class KEY:

    MODEL_NAME = "model_name"
    REQUEST_TYPE = "request_type"
    PREDICT_METHOD = "predict_method"
    RESPONSE_STATUS = "status"
    ERROR_INFO = "errinfo"

    class PREDICTION:
        IMAGE_NUMS = "image_nums"
        MEMORY_ADDR = "address"     # 图片地址（共享内存）
        SCALE = "scale"             #
        COLS = "cols"               # 图片col
        ROWS = "rows"               # 图片row
        CHANNELS = "channels"       # 图片channel
        KEYPS = "keypoints"         # 是否要关键点返回
        MASK = "mask"               # 是否要MASK的返回

        CLASSES = "classes"
        CONFIDENCES = "confidences"
        BOXES = "boxes"
        BOXMODE = "boxmode"
        KEYPSINFO = "keypinfo"
        MASKINFO = "maskinfo"


# 值
class VALUE:

    class REQUEST_TYPE:
        MODEL_INIT = 0
        READ_BUFFER = 1
        MODEL_PREDICT = 2
        THREAD_CLOSED = 3
    #
    # class PREDICT_METHOD:
    #     MEMORY = 0
    #     SOCKET = 1
    #     HTTP = 2

    class RESPONSE_STATUS:
        SUCCEED = 0
        FAILED = 1
        CLOSED = 2

    class BOXMODE:
        XYXY = 0
        XYWH = 1
        XYXY_PERCENTAGE = 2
        XYWH_PERCENTAGE = 3

    class MASK:
        FALSE = 0
        TRUE = 1

    class KEYPS:
        FALSE = 0
        TRUE = 1


# format
failure_example_init = {
    KEY.REQUEST_TYPE: VALUE.REQUEST_TYPE.MODEL_INIT,
    KEY.RESPONSE_STATUS: VALUE.RESPONSE_STATUS.FAILED,
    KEY.ERROR_INFO: ""
}

failure_example_predict = {
    KEY.REQUEST_TYPE: VALUE.REQUEST_TYPE.MODEL_PREDICT,
    KEY.RESPONSE_STATUS: VALUE.RESPONSE_STATUS.FAILED,
    KEY.ERROR_INFO: ""
}
success_example_init = {
    KEY.REQUEST_TYPE: VALUE.REQUEST_TYPE.MODEL_INIT,
    KEY.RESPONSE_STATUS: VALUE.RESPONSE_STATUS.SUCCEED,
}

success_example_buffer = {
    KEY.REQUEST_TYPE: VALUE.REQUEST_TYPE.READ_BUFFER,
    KEY.RESPONSE_STATUS: VALUE.RESPONSE_STATUS.SUCCEED
}

success_example_predict = {
    KEY.REQUEST_TYPE: VALUE.REQUEST_TYPE.MODEL_PREDICT,
    KEY.RESPONSE_STATUS: VALUE.RESPONSE_STATUS.SUCCEED,
    KEY.PREDICTION.CLASSES: [],
    KEY.PREDICTION.CONFIDENCES: [],
    KEY.PREDICTION.BOXES: [],
    KEY.PREDICTION.BOXMODE: VALUE.BOXMODE.XYXY,
    KEY.PREDICTION.KEYPS: [],
    KEY.PREDICTION.MASKINFO: []
}

closed_example = {
    KEY.REQUEST_TYPE: VALUE.REQUEST_TYPE.THREAD_CLOSED,
    KEY.RESPONSE_STATUS: VALUE.RESPONSE_STATUS.CLOSED,
}
