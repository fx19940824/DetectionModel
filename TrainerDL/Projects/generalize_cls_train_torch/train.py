import os
import re
import time
import torch
import numpy as np
from tqdm import tqdm
from argparse import Namespace
from torchsummary import summary
from Utils.parsers import parse_cfg
from Utils.loss import soft_cross_entropy
from Utils.label_utils import one_hot_encoding, label_smoothing
from Utils.attacks.fast_gradient_sign_targeted import FastGradientSignTargeted
from classifications.utils import model_factory
from Projects.generalize_cls_train_torch.data_loader import data_loader


class Generalization_CLS:
    def __init__(self, train_cfg):
        cfg = parse_cfg(train_cfg)
        cfg.update(parse_cfg(cfg["cfgpath"]))
        self.args = Namespace(**cfg)
        self.print_option()
        self.dataloader = data_loader(self.args)
        self.epoch = 0
        self.model = model_factory.initialize_model(self.args.modelname, self.args.classes, feature_extract=False, use_pretrained=None)
        self.model.cuda()
        if self.args.init_weight:
            state_dict = torch.load(self.args.init_weight)
            if 'epoch' in state_dict:
                self.epoch = state_dict['epoch']
                self.model.load_state_dict(torch.load(state_dict['net']))
            elif isinstance(state_dict, dict):
                self.load_pretrain_model(state_dict)

        self.criterion = soft_cross_entropy()
        self.build_prerequisite()

    def build_prerequisite(self):
        # froze layers and set different lr for layers
        if self.args.transfer_learning:
            finetune_params = []
            frozen_match = r'|'.join(self.args.frozen_layers if isinstance(self.args.frozen_layers, list) else [self.args.frozen_layers])
            finetune_match = r'|'.join(self.args.finetune_layers if isinstance(self.args.finetune_layers, list) else [self.args.finetune_layers])

            for param in self.model.named_parameters():
                if re.match(frozen_match, param[0]):
                    param[1].requires_grad = False
                if re.match(finetune_match, param[0]):
                    finetune_params.append(id(param[1]))

            base_params = filter(lambda p: id(p) not in finetune_params,
                                 self.model.parameters())

            finetune_params_filter = filter(lambda p: id(p) in finetune_params,
                                     self.model.parameters())

            self.optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': finetune_params_filter, 'lr': self.args.finetune_rate * self.args.lr},
            ], lr=self.args.lr, weight_decay=1e-4)

        else:
            self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.args.lr)

        if self.args.attack:
            self.fgst = FastGradientSignTargeted(self.args.attack_eps / 255, loss_func=self.criterion)
        else:
            self.fgst = None

        self.optimizer.zero_grad()

    def batch_process(self, imgs, lbls):
        oh_lbls = one_hot_encoding(lbls, self.args.classes)

        if self.args.label_smoothing:
            oh_lbls = label_smoothing(oh_lbls, self.args.classes, self.args.label_smoothing_eps)

        if self.args.mix_up:
            lam = np.random.beta(self.args.alpha, self.args.alpha)
            imgs = torch.autograd.Variable(
                lam * imgs[:self.args.batchsize, :] + (1. - lam) * imgs[self.args.batchsize:, :])
            oh_lbls = torch.autograd.Variable(
                lam * oh_lbls[:self.args.batchsize, :] + (1. - lam) * oh_lbls[self.args.batchsize:, :])

        imgs = imgs.cuda()
        oh_lbls = oh_lbls.cuda()

        if self.args.attack and np.random.rand(1) < self.args.attack_rate:
            self.model.eval()
            imgs = self.fgst.generate(self.model, imgs, 1 - oh_lbls)
            self.model.train()

        return imgs, oh_lbls

    def train_batch(self, batch, train_count, train_loss, train_acc):
        lbls = batch[1].cuda()
        imgs, oh_lbls = self.batch_process(batch[0], batch[1])
        outs = self.model(imgs)
        loss = self.criterion(outs, oh_lbls)    #oh_lbls: one_hot encoding标签，lbls: 默认标签，注意使用的损失函数要求的输入是什么
        train_count += batch[0].shape[0]
        train_loss += loss.data

        pred = torch.max(outs, 1)[1]
        train_correct = (pred == lbls).sum()
        train_acc += train_correct.data

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return train_count, train_loss, train_acc

    def train_epoch(self):
        train_loss = 0.
        train_acc = 0.
        train_count = 0.
        self.model.train()
        for iteration, batch in tqdm(enumerate(self.dataloader['train']),
                                            leave=False, total=len(self.dataloader['train'])):
            train_count, train_loss, train_acc = \
                self.train_batch(batch, train_count, train_loss, train_acc)
        print('Train  Avg Loss: {:.6f}, Acc: {:.6f}'.format(
            train_loss / len(self.dataloader['train']),
            train_acc.float() / train_count))

    def eval(self, stage):
        self.model.eval()
        eval_loss = 0.
        eval_acc = 0.
        eval_count = 0.
        torch.cuda.empty_cache()
        for imgs, lbls in tqdm(self.dataloader[stage], leave=False, total=len(self.dataloader[stage])):
            oh_lbls = one_hot_encoding(lbls, self.args.classes).cuda()
            lbls = lbls.cuda()
            imgs = imgs.cuda()
            eval_count += imgs.shape[0]
            outs = self.model(imgs)
            loss = self.criterion(outs, oh_lbls)
            eval_loss += loss.data
            pred = torch.max(outs, 1)[1]
            num_correct = (pred == lbls).sum()
            eval_acc += num_correct.data
        print('{}  Avg Loss: {:.6f}, Acc: {:.6f}'.format(
            stage, eval_loss / (len(self.dataloader[stage])),
            eval_acc.float() / eval_count))
        time.sleep(1)

    def save_weight(self):
        if self.epoch < self.args.epochs:
            torch.save({'net': self.model.state_dict(), 'epoch': self.epoch + 1}, os.path.join(self.args.weight_out, 'model_epoch_{}.pth'.format(self.epoch+1)))
            print('save weights ...')
        else:
            torch.save(self.model.state_dict(), os.path.join(self.args.weight_out, 'model_final.pth'))

    def train(self):
        summary(self.model, (3, 224, 224), (1))
        for epoch in range(self.epoch, self.args.epochs):
            print('epoch {} :train'.format(epoch + 1))
            self.train_epoch()
            with torch.no_grad():
                print('epoch {} :{}'.format(epoch + 1, "test"))
                self.eval("test")

                print('epoch {} :{}'.format(epoch + 1, "valid"))
                self.eval("valid")
            self.epoch += 1
            self.save_weight()

    def print_option(self):
        print('hyperparams ')
        for hyperparam in self.args.__dict__.items():
            print(hyperparam[0], ': ', hyperparam[1])

    def load_pretrain_model(self, state_dict):
        for name, param in self.model.named_parameters():
            if name in state_dict:
                if state_dict[name].data.shape == param.data.shape:
                    param.data = state_dict[name].data.cuda().float()

if __name__ == '__main__':
    gac = Generalization_CLS("cfgs/train_example_cls.cfg")
    gac.train()


