import torch
from Utils.parsers import parse_cfg
from argparse import Namespace
from torchsummary import summary
from Projects.BM_generalization.data.data_loader import data_loader
from Projects.BM_generalization.models.model import get_model
from Utils.analysis.mmd import *

'''
    域适应中衡量目标域与源域间距离的example
'''
domain = ["train", "test", "valid"]
def get_features(train_cfg):

    args = parse_cfg(train_cfg)
    args.update(parse_cfg(args["cfgpath"]))
    args = Namespace(**args)
    dataloader = data_loader(args, is_Train=False)
    model = get_model(args.modelname, args.classes, freeze_layer=False, weights=args.init_weight)
    model.cuda()
    model.eval()
    summary(model, (3, 224, 224))
    total_batch = min(len(dataloader["train"]), len(dataloader["test"]), len(dataloader["valid"])) - 1
    features = {}

    with torch.no_grad():
        for stage in domain:
            feature = []
            for it, (imgs, _) in enumerate(dataloader[stage]):
                imgs = imgs.cuda()
                outs = model.features(imgs)
                outs = model.global_pool(outs)
                feature.append(outs)
                if it > total_batch:
                    break
            features[stage] = torch.cat(feature, dim=0).squeeze()


    return features, total_batch

features , total_batch = get_features("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/bomo_generalize_torch/cfgs/train_bomo_cls.cfg")

features_another, _ = get_features("/home/cobot/caid2.0/python/Main/TrainerDL/Projects/bomo_generalize_torch/cfgs/train_bomo_cls.cfg")
batchsize = 64

for i, stage1 in enumerate(domain):
    # losses_itself = []
    # for l in range(0, total_batch * batchsize, batchsize):
    #     losses_itself.append(float(mmd_rbf(features[stage1][l:l + batchsize], features_another[stage1][l:l + batchsize]).cpu().numpy()))
    #
    # print("self compare: %s vs %s" % (stage1, stage1), sum(losses_itself) / len(losses_itself))

    for j, stage2 in enumerate(domain[i+1:]):
        losses = []
        for k in range(0, total_batch * batchsize, batchsize):
            losses.append(float(mmd_rbf(features[stage1][k:k+batchsize], features[stage2][k:k+batchsize]).cpu().numpy()))

        print("cross_domain compare: %s vs %s" % (stage1, stage2), sum(losses)/len(losses))



pass