import torch


def linear(x, weight, bias=None):
    out = torch.bmm(x, weight.transpose(-2, -1))

    if bias is not None:
        out = out + bias.unsqueeze(-2)

    return out
