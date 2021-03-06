import torch
from torchvision.models import resnet
from Utils.PytorchToCaffe import pytorch_analyser

if __name__ == '__main__':
    resnet18 = resnet.resnet18()
    input_tensor = torch.ones(1, 3, 224, 224)
    blob_dict, tracked_layers = pytorch_analyser.analyse(resnet18, input_tensor)
    pytorch_analyser.save_csv(tracked_layers, '/tmp/analysis.csv')

