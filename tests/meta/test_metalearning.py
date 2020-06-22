import pytest
from torch import nn

from bestiary.meta.collections.sinusoids import Sinusoids
from bestiary.nets.meta import MetaLearningNet


@pytest.fixture
def sinusoids():
    return Sinusoids(100)


class Module(nn.Module):

    def __init__(self):
        super(Module, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, support, query):
        x, y, length = support
        return self.net(x)


def disable_callbacks(net):
    callbacks = [
        k for k in net.get_params().keys()
        if k.startswith('callbacks__') and len(k.split('__')) == 2
    ]
    net.set_params(
        **{cb: None for cb in callbacks}
    )


def test_net(sinusoids):
    net = MetaLearningNet(Module, criterion=nn.MSELoss)
    disable_callbacks(net)
    # noinspection PyTypeChecker
    net.fit(sinusoids, epochs=1)
