from contextlib import contextmanager

from torch import nn


def recursive_duplicate(module, t):
    for child in module.children():
        recursive_duplicate(child, t)

    if isinstance(module, MetaMixin):
        module.duplicate(t)

    return module


def recursive_flush(module):
    for child in module.children():
        recursive_flush(child)

    if isinstance(module, MetaMixin):
        module.flush()

    return module


class MetaMixin(object):

    def children(self):
        pass

    def flush(self):
        pass

    def duplicate(self, t: int):
        pass

    @contextmanager
    def replicate(self, t: int) -> nn.Module:
        recursive_duplicate(self, t)
        yield self
        recursive_flush(self)


class MetaModule(nn.Module, MetaMixin):

    def __init__(self):
        super(MetaModule, self).__init__()

    def flush(self):
        pass

    def duplicate(self, t: int):
        pass

    @contextmanager
    def replicate(self, t: int) -> nn.Module:
        recursive_duplicate(self, t)
        yield self
        recursive_flush(self)
