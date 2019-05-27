import torch
from Utils.parsers import parse_cfg
from argparse import Namespace
from torchsummary import summary
from Projects.BM_generalization.data.data_loader import data_loader
from Projects.BM_generalization.models.model import Model
from Utils.analysis.mmd import *

domain = ["train", "test", "valid"]
labels = ["BM", "ZC"]
def get_features(train_cfg):

    args = parse_cfg(train_cfg)
    args.update(parse_cfg(args["cfgpath"]))
    args = Namespace(**args)
    dataloader = data_loader(args, is_Train=False)
    model = Model(args.modelname, args.classes, freeze_layer=False, weights=args.init_weight).bn_model

    model.load_state_dict(torch.load(args.init_weight))
    model.cuda()
    model.eval()
    summary(model, (3, 224, 224))
    total_batch = min(len(dataloader["train"]), len(dataloader["test"]), len(dataloader["valid"])) - 1
    features = {}

    with torch.no_grad():
        for stage in domain:
            feature = {0: [], 1: []}
            for it, (imgs, lbls) in enumerate(dataloader[stage]):
                imgs = imgs.cuda()
                outs = model.features(imgs)
                outs = model.global_pool(outs)
                lbls = lbls.numpy().tolist()
                for idx, lbl in enumerate(lbls):
                    feature[lbl].append(outs[idx].unsqueeze(dim=0))

                if it > total_batch:
                    break
            feature[0] = torch.cat(feature[0], dim=0).squeeze()
            feature[1] = torch.cat(feature[1], dim=0).squeeze()
            features[stage] = feature


    return features, total_batch

features , total_batch = get_features("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/generalize_torch/train_bomo_cls.cfg")
batchsize = 64

for i, stage1 in enumerate(domain):
    for j, stage2 in enumerate(domain[i+1:]):
        for lb in [0, 1]:
            losses = []
            for k in range(0, total_batch * batchsize, batchsize):
                dm1 = features[stage1][lb][k:k+batchsize]
                dm2 = features[stage2][lb][k:k+batchsize]
                if dm1.shape[0] != dm2.shape[0]:
                    break
                losses.append(float(mmd_rbf(dm1, dm2).cpu().numpy()))

            print("%s cross_domain compare: %s vs %s  " % (labels[lb], stage1, stage2,), sum(losses)/len(losses))



pass