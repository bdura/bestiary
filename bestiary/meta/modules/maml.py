from typing import Tuple

import torch

from bestiary.meta.modules.base import MetaModule


class MAML(MetaModule):

    def __init__(self, module, lr=1e-3, **kwargs):
        super(MAML, self).__init__()

        self.network = module(**kwargs)

        self.lr = lr

    def forward(self, support: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                query: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        xs, ys, ls = support
        xq, lq = query

        with self.network.adapt(xs, ys, ls) as net:
            return net(xq, lq)

    def adapt(
            self,
            support: torch.Tensor,
            targets: torch.Tensor,
            lengths: torch.Tensor,
    ) -> torch.Tensor:
        pass
