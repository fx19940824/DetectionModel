from classifications.utils import model_factory
import torch
from torchsummary import summary
model = model_factory.initialize_model("xception", 2, feature_extract=True, use_pretrained=None).cuda()
# model.load_state_dict(torch.load("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/FLL_flaw_detection/classification/weights/model_final.pth"))

# example = torch.rand(1, 3, 299, 299).cuda().half()
summary(model, (3, 299, 299), (1))
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
model.eval()
# traced_script_module = torch.jit.trace(model, example)
# output = traced_script_module(torch.ones(2, 3, 299, 299).cuda().half())
# traced_script_module.save("xception.pt")

# paths = [
#     "/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_分类 /test/good/2019.02.27_09:52:29.227_7f731c0009c0_12.png",
#     "/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_分类 /test/good/2019.02.27_09:52:29.227_7f731c0009c0_22.png",
#     "/media/cobot/7dbae065-b360-4e5b-9f19-63aecce47d84/FLL_分类 /test/good/2019.02.27_09:52:29.227_7f731c0009c0_38.png",
# ]
# import cv2
# import numpy as np
# imgs = [cv2.imread(img) for img in paths]
# imgs = [cv2.resize(img, (299, 299)) for img in imgs]
# imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
# imgs = np.array([img.astype(np.float32) / 255.0 for img in imgs])
# model = torch.jit.load("xception.pt").cuda()
# # test = torch.ones(2, 3, 299, 299).cuda()
# test = torch.Tensor(imgs.transpose([0,3,1,2])).cuda()
# print(model(test))