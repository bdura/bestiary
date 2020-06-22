from typing import Tuple

import torch
from torch import nn


class MetaModule(nn.Module):

    def __init__(self):
        super(MetaModule, self).__init__()

    def forward(self, support: Tuple[torch.Tensor], query: Tuple[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def adapt(
            self,
            support: Tuple[torch.Tensor],
            targets: Tuple[torch.Tensor],
            lengths: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError
