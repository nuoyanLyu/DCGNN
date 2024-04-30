import torch
from torch import nn


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # 报错，MPS向量不允许float64也就是double类型，更改为float32类型
    correct = preds.eq(labels).to(torch.float32)
    correct = correct.sum()
    return correct / len(labels)


def init_weights(m):
    if type(m) == nn.Linear:
        # change another init method
        nn.init.kaiming_uniform_(m.weight)
        # nn.init.xavier_uniform_(m.weight)
