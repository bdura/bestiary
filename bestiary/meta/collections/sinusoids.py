import json
from typing import Tuple, Dict, List

import numpy as np

from bestiary.meta.dataset import MetaDataset
from bestiary.utils.data import RandomDataset


class Sinusoid(RandomDataset):

    def __init__(self, amplitude: float, frequency: float, phase: float,
                 noise: float = 0., limits: Tuple[float] = (-5, 5)):
        r"""
        Random dataset that outputs a sinusoid.

        Parameters
        ----------
        amplitude: The amplitude of the sinusoid.
        frequency: The frequency.
        phase: The initial phase.
        noise: The standard deviation :math:`\sigma`
        limits: The sampling limits for :math:`x`.
        """

        self.limits_ = limits

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

        self.noise = noise

    def __len__(self):
        return 2 ** 32

    def sample_x(self):
        return np.random.uniform(*self.limits_)

    def __getitem__(self, item):
        x = self.sample_x()
        y = self.amplitude * np.sin(self.frequency * x + self.phase) + np.random.normal(0, self.noise)

        x, y = np.float32([x]), np.float32([y])

        return x, y


def sample_characteristics():
    return dict(
        amplitude=np.random.uniform(1, 5),
        frequency=np.random.uniform(1 / 10, 5 / 10) * 2 * np.pi,
        phase=np.random.uniform(-1, 1) * np.pi,
    )


class Sinusoids(MetaDataset):

    def __init__(self, classes: int = 10000, noise: float = 0., shots: int = 10,
                 characteristics: List[Dict] = None):
        if characteristics is None:
            characteristics = [sample_characteristics() for _ in range(classes)]

        self.characteristics_ = dict(characteristics=characteristics, noise=noise, shots=shots)

        datasets = [
            Sinusoid(**characteristic, noise=noise)
            for characteristic in characteristics
        ]
        super().__init__(datasets=datasets, shots=shots)

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.characteristics_, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            characteristics = json.load(f)

        return cls(**characteristics)
