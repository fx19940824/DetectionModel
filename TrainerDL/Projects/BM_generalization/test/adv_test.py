
from Projects.BM_generalization.models.model import Model
from Utils.attacks.fast_gradient_sign_untargeted import FastGradientSignUntargeted
from Utils.attacks.fast_gradient_sign_targeted import FastGradientSignTargeted
from torchvision.transforms import functional as TF
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Projects.BM_generalization.data.data_loader import data_loader
from Projects.BM_generalization.train import Great_augment_cls
from PIL import Image
from Utils.parsers import parse_cfg
from argparse import Namespace
import torch
import numpy as np


def preprocess(x):
    if isinstance(x, str):
        x = TF.Image.open(x)
    x = TF.resize(x, 299)
    x = TF.to_tensor(x).unsqueeze(0)
    x = Variable(x, requires_grad=True)
    return x

def recreate(x):
    x = x.squeeze()
    x = TF.to_pil_image(x)
    return x
# train_transform = transforms.Compose([
#     # transforms.Lambda(lambda x: np.array(x, dtype=np.uint8)),
#     # transforms.Lambda(lambda x: aug.augment_image(x)),
#     # transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     # transforms.RandomApply([transforms.RandomAffine(degrees=5, translate=(0.01, 0.01), scale=(0.95, 1.05))], p=0.3),
#     transforms.Resize(299),
#     transforms.CenterCrop(299),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

args = parse_cfg("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/generalize_torch/cfgs/train_bomo_cls.cfg")
args.update(parse_cfg(args["cfgpath"]))
args = Namespace(**args)
Gac = Great_augment_cls("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/generalize_torch/cfgs/train_bomo_cls.cfg")

model = Gac.model.backbone
model.load_state_dict(torch.load("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/generalize_torch/weights/best_ever869.pth"))


dataloader = data_loader(args)
fgst = FastGradientSignTargeted(0.3/255)
model.cuda()
cur = 0
for imgs, lbls in dataloader["valid"]:
    model.eval()
    imgs = imgs.cuda()
    #可视化

    or_img = imgs[0].cpu()
    or_img = TF.to_pil_image(or_img)
    plt.figure("ori")
    plt.imshow(or_img)
    plt.show()
    # 原标签

    outs = model(imgs)
    print(outs[0])
    pred = torch.max(outs, 1)[1]
    print(pred[0])


    # 生成样本
    imgs, lbls = Gac.batch_process(imgs, lbls)
    # 测试对抗样本
    outs = model(imgs)
    print(outs[0])
    pred = torch.max(outs, 1)[1]
    print(pred[0])
    # 可视化
    img = imgs[0].cpu()
    img = TF.to_pil_image(img)
    plt.figure("adv")
    plt.imshow(img)
    plt.show()
    noise = np.array(or_img) - np.array(img)
    plt.figure("noise")
    plt.imshow(noise)
    plt.show()
    pass





