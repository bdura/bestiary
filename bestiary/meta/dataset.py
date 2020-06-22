from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from bestiary.utils.data import DatasetSplitter


class InfiniteDataLoader(object):

    def __init__(self, dataset, **kwargs):
        self.dataloader = DataLoader(dataset, **kwargs)

    def __iter__(self):
        iterator = iter(self.dataloader)

        # Iterate over dataset
        while True:
            try:
                data = next(iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # In that case, reinitialise data loader
                iterator = iter(self.dataloader)
                data = next(iterator)
            yield data


class MetaDataset(object):

    def __init__(self, datasets: List[Dataset], shots: int = 10, ways: int = 1, **kwargs):

        self.datasets = datasets

        splitter = DatasetSplitter(split=.2)

        self.support_, self.query_ = list(zip(*[splitter(dataset) for dataset in datasets]))

        # self.shots = shots
        self.length_ = ways * shots

        # self.iterators_ = [
        #     iter(InfiniteDataLoader(dataset, batch_size=shots, **kwargs))
        #     for dataset in datasets
        # ]

        self.support_iterators_ = [
            iter(InfiniteDataLoader(dataset, batch_size=shots, **kwargs))
            for dataset in self.support_
        ]

        self.query_iterators_ = [
            iter(InfiniteDataLoader(dataset, batch_size=shots, **kwargs))
            for dataset in self.query_
        ]

    def split_dataset_(self, dataset):
        pass

    def __len__(self):
        return len(self.datasets)

    def pack_data(self, data):

        if isinstance(data, list):
            x, y = data
        else:
            x, y = data, None

        length = len(x)

        if length < self.length_:
            x = torch.cat((
                x,
                x.new_zeros((self.length_ - length, *x.shape[1:])),
            ))

            if y is not None:
                y = torch.cat((
                    y,
                    y.new_zeros((self.length_ - length, *y.shape[1:])),
                ))

        if y is None:
            return x, length

        else:
            return x, y, length

    def __getitem__(self, item):
        support_iterator = self.support_iterators_[item]
        query_iterator = self.query_iterators_[item]
        support, query = next(support_iterator), next(query_iterator)
        return dict(support=self.pack_data(support), query=self.pack_data(query))
