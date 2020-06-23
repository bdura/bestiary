import torch
from torch import nn
from torch.nn import functional

import bestiary.meta.layers.functional as mf
from bestiary.meta.layers.module import MetaMixin


class Linear(nn.Linear, MetaMixin):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

        self.replicated_weight = None
        self.replicated_bias = None

    def forward(self, x):
        if self.replicated_weight is not None:
            return mf.linear(x, self.replicated_weight, self.replicated_bias)

        else:
            return functional.linear(x, self.weight, self.bias)

    def duplicate(self, t: int):
        self.replicated_weight = torch.stack([self.weight for _ in range(t)])
        if self.bias is not None:
            self.replicated_bias = torch.stack([self.bias for _ in range(t)])

    def flush(self):
        self.replicated_weight = None
        self.replicated_bias = None
