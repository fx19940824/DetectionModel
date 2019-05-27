from ServerDL.cfgs.Protocol import *
import json


class MODEL_INIT_ERROR:
    MODEL_LOAD_EXCEPTION = KeyError
    GPU_MEMORY_EXCEPTION = 2


class MODEL_PREDICT_ERROR:
    MODEL_EXIST_EXCEPTION = UnboundLocalError
    IMAGE_FORMAT_EXCEPTION = 3
    IMAGE_EMPTY_EXCEPTION = 4
    IMAGE_SHAPE_EXCEPTION = 5


def ExceptionSendBack(exception, sock, client_type):
    result = {}
    string = "stage " + str(client_type) + '   ' + exception.__class__.__name__ + ':' + ''.join(exception.args)

    result[KEY.REQUEST_TYPE] = client_type
    result[KEY.RESPONSE_STATUS] = VALUE.RESPONSE_STATUS.FAILED
    result[KEY.ERROR_INFO] = string

    sock.send(json.dumps(result).encode('utf-8'))


# print(MODEL_INIT_ERROR.MODEL_LOAD_EXCEPTION.value)

def HandleException(exception, client_type):
    print("\033[0;31m %s \033[0m!" % ("stage " + str(client_type) + '   ' + exception.__class__.__name__ + ':' + ''.join(exception.args)))