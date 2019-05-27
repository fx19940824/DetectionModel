import torch


def one_hot_encoding(label, class_num):
    label = torch.unsqueeze(label, 1)
    one_hot = torch.zeros(label.shape[0], class_num).scatter_(1, label, 1)
    return one_hot


def label_smoothing(one_hot_labels, class_num, eps=0.1):
    return (1.0 - eps) * one_hot_labels + eps / class_num
