import torch
from Utils.parsers import parse_cfg
from argparse import Namespace
import cv2
import numpy as np
from Projects.BM_generalization.data.data_loader import data_loader
from Projects.BM_generalization.models.model import Model
from Utils.analysis.mmd import *
from tqdm import  tqdm
from torchvision.transforms import functional as TF

train_cfg = "/home/cobot/caid2.0/python/Main/TrainerDL/Projects/generalize_torch/cfgs/train_bomo_cls.cfg"
args = parse_cfg(train_cfg)
args.update(parse_cfg(args["cfgpath"]))
args = Namespace(**args)
dataloader = data_loader(args, is_Train=False)
model = Model(args.modelname, args.classes, freeze_layer=False, weights=args.init_weight).backbone

weight = torch.load(args.init_weight)
model.load_state_dict(weight)
index = []
model.cuda()
model.eval()
d = dataloader["valid"].dataset.imgs
for idx, (imgs, lbls) in tqdm(enumerate(dataloader["valid"]), leave=False, total=len(dataloader['valid'])):
    imgs = imgs.cuda()
    preds = model(imgs).cpu()
    preds = torch.max(preds, 1)[1]
    err = lbls != preds
    err = err.cpu().numpy()
    err_idx = np.where(err == 1)[0]
    index.extend([erridx + idx*64 for erridx in err_idx])
print(len(index)/(len(dataloader["valid"])*64))
O = []
I = []
for idx in index:
    if d[idx][1] == 0:
        O.append(d[idx][0])
    else:
        I.append(d[idx][0])
open("BM.txt", 'w').write('\n'.join(O))
open("ZC.txt", 'w').write('\n'.join(I))