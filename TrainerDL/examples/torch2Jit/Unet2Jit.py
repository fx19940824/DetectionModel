import torch
import torchvision
from Projects.FLL_flaw_detection.segmentation.Unet.unet.unet_model import UNet
from torchsummary import summary
model = UNet(3, 1).cuda()
# model.load_state_dict(torch.load("/home/cobot/caid2.0/python/Main/TrainerDL/weights/CP3.pth"))
# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224).cuda()
summary(model, (3, 224, 224), (1))
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
model.eval()
traced_script_module = torch.jit.trace(model, example)
output = traced_script_module(torch.ones(1, 3, 224, 224).cuda())
# traced_script_module.save("unet.pt")