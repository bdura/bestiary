import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from bestiary.meta.dataset import InfiniteDataLoader, MetaDataset


class DummyDataset(Dataset):

    def __init__(self, n):
        super(DummyDataset, self).__init__()

        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return torch.tensor([item]), torch.tensor([1])


@pytest.fixture
def dataset():
    return DummyDataset(10)


def test_infinite_iterator(dataset):
    dataloader = DataLoader(dataset, batch_size=5)

    with pytest.raises(StopIteration):
        iterator = iter(dataloader)
        for _ in range(10):
            next(iterator)

    infinite_dataloader = InfiniteDataLoader(dataset, batch_size=5)
    iterator = iter(infinite_dataloader)
    for _ in range(10):
        next(iterator)


def test_metadataset(dataset):
    metadataset = MetaDataset([dataset for _ in range(5)], shots=10)

    item = metadataset[0]
    assert isinstance(item, dict)
    assert 'support' in item and 'query' in item and len(item) == 2
