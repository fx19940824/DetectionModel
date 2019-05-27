from classifications.utils import model_factory
from torchvision.models import inception_v3
from torchvision.models.resnet import *
from Utils.layers.L_Softmax import LSoftmaxLinear
import torch

def get_model(name="bninception", num_classes=2, freeze_layer=True, weights=None):
    return model_factory.initialize_model(name, num_classes, freeze_layer, weights)