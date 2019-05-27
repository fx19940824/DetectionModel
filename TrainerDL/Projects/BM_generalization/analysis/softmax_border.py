import torch
from Utils.parsers import parse_cfg
from argparse import Namespace
from torchsummary import summary
from Projects.BM_generalization.data.data_loader import data_loader
from Projects.BM_generalization.models.model import Model
from Utils.analysis.mmd import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

domain = ["test", "valid"]
def get_features(train_cfg):

    args = parse_cfg(train_cfg)
    args.update(parse_cfg(args["cfgpath"]))
    args = Namespace(**args)
    dataloader = data_loader(args, is_Train=False)
    model = Model(args.modelname, args.classes, freeze_layer=False, weights=args.init_weight, l_softmax=None)
    weights = torch.load("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/generalize_torch/weights/model_epoch_9.pth")

    model.load_state_dict(weights)
    model.cuda()
    model.eval()
    # summary(model, (3, 224, 224))
    softmax = torch.nn.LogSoftmax(dim=-1)
    features = {}



    with torch.no_grad():
        for stage in domain:
            feature = {0: [], 1: []}
            for it, (imgs, lbls) in enumerate(dataloader[stage]):
                imgs = imgs.cuda()
                outs = model(imgs)
                # outs = softmax(preds)
                lbls = lbls.numpy().tolist()
                for idx, lbl in enumerate(lbls):
                    feature[lbl].append(outs[idx].unsqueeze(dim=0))
            feature[0] = torch.cat(feature[0], dim=0).cpu().numpy()
            feature[1] = torch.cat(feature[1], dim=0).cpu().numpy()
            features[stage] = feature
    return features


features = get_features("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/generalize_torch/cfgs/train_bomo_cls.cfg")
fig = plt.figure()
axes = fig.add_subplot(111)
plt.xlim((-6, 6))
plt.ylim((-6, 6))

maker = {0: ".", 1: "^"}
test_colors = ["red", "blue"]
valid_colors = ["blue",  "green"]

for lbl, points in features["valid"].items():
    axes.scatter(points[:, 0], points[:, 1], color=test_colors[lbl], s=1, marker=maker[0])
# for lbl, points in features["valid"].items():
#     axes.scatter(points[:4000, 0], points[:4000, 1], color='blue', s=1, marker=maker[0])

# i = 0
# for lbl, points in features["test"].items():
#     if i > 2000:
#         break
#     axes.scatter(points[:, 0], points[:, 1], color='yellow', s=1, marker=maker[lbl])
#     i += 1

plt.xlabel('X1')
plt.ylabel('x2')
plt.title('Epic Chart')
plt.show()
pass
