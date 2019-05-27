import sys
sys.path.insert(0, '.')
import torch
from torch.autograd import Variable
from torchvision.models.inception import inception_v3
from Utils.PytorchToCaffe import pytorch_to_caffe
from classifications.utils.model_factory import initialize_model

if __name__=='__main__':
    name = 'bninception'
    net = initialize_model(model_name=name, num_classes=2, feature_extract=False, use_pretrained=None)

    weight = torch.load("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/generalize_torch/weights/model_final_1.pth")
    net.load_state_dict(weight)
    net.eval()

    input = torch.ones([1,3,224,224])
    pytorch_to_caffe.trans_net(net, input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))