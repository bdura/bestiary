from torch import nn

from bestiary.meta.layers.module import MetaModule, recursive_meta_in, recursive_meta_out


class DummyModule(MetaModule):

    def __init__(self):
        super(DummyModule, self).__init__()
        self.t = 1

    def meta_in(self, t: int):
        self.t = t

    def meta_out(self):
        self.t = 1


def test_replication():
    module = DummyModule()

    assert module.t == 1
    with module.meta(10):
        assert module.t == 10
    assert module.t == 1


def test_sequential():
    module = nn.Sequential(
        DummyModule(),
        nn.Linear(4, 5),
        DummyModule()
    )

    dummy1, dummy2 = list(module.children())[0], list(module.children())[-1]

    assert dummy1.t == 1 and dummy2.t == 1
    recursive_meta_in(module, 5)
    assert dummy1.t == 5 and dummy2.t == 5
    recursive_meta_out(module)
    assert dummy1.t == 1 and dummy2.t == 1
