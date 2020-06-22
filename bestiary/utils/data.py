from typing import Union

import numpy as np
import torch
from skorch.dataset import CVSplit
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    """
    A random dataset can be replicated to get a clean split.
    """

    @property
    def valid(self):
        return self


class DatasetSplitter(object):

    def __init__(self, split: Union[int, float] = .8, minimum_train: int = 10):

        self.split = split
        self.minimum_valid = minimum_train

    def __call__(self, dataset: Union[Dataset, np.array],
                 y: Union[torch.Tensor, np.array, None] = None):

        valid = getattr(dataset, 'valid', None)

        if valid is not None:
            return dataset, valid

        if isinstance(self.split, CVSplit):
            train, valid = self.split(dataset, y)

        else:
            split = CVSplit(self.split)
            train, valid = split(dataset, y)

        return train, valid
