from itertools import product

import numpy as np


class SourdoughEnvironment(object):

    def __init__(self, size: int = 100, energy: float = .2, beacon: int = 10):

        self.size = size
        self.beacon = beacon

        self.offset = 2

        self.beacon_position = np.random.choice([int(beacon * self.offset), size - int(beacon * self.offset)], size=2,
                                                replace=True)
        self.energy = energy

        self.mask_ = None

        self.directions_ = {
            str(np.array([int(beacon * self.offset), int(beacon * self.offset)])): np.array([1, 0]),
            str(np.array([int(beacon * self.offset), size - int(beacon * self.offset)])): np.array([0, -1]),
            str(np.array([size - int(beacon * self.offset), int(beacon * self.offset)])): np.array([0, 1]),
            str(np.array([size - int(beacon * self.offset), size - int(beacon * self.offset)])): np.array([-1, 0])
        }
        self.direction_ = self.directions_[str(self.beacon_position)]

    @property
    def mask(self):
        if self.mask_ is None:
            self.mask_ = np.zeros((2 * self.beacon + 1, 2 * self.beacon + 1))
            for i, j in product(range(2 * self.beacon + 1), range(2 * self.beacon + 1)):
                if np.sqrt((i - self.beacon) ** 2 + (j - self.beacon) ** 2) < self.beacon:
                    self.mask_[i, j] = 1
        return self.mask_

    def step(self):

        self.direction_ = self.directions_.get(str(self.beacon_position), self.direction_)

        self.beacon_position += self.direction_
        # self.beacon_position = self.beacon_position.clip(self.beacon, self.size - self.beacon - 1)

        energy = np.zeros((self.size, self.size))

        x, y = self.beacon_position
        energy[x - self.beacon:x + self.beacon + 1, y - self.beacon:y + self.beacon + 1] = self.energy * self.mask

        return energy

    def reset(self):
        self.beacon_position = (self.size // 2, self.size // 2)


def select_index(array: np.ndarray, indices: np.ndarray):
    """

    Parameters
    ----------
    array: Array to select from.
        Must be reshaped such that the dimensions are indexed in the right order.
    indices: The indices to select.

    Returns
    -------
    out: The selected array.
    """

    indices = indices.astype(np.int).reshape(len(indices), -1)

    shape = array.shape[indices.shape[1]:]

    out = np.empty((len(indices), *shape))

    for i, selection in enumerate(indices):
        out[i] = array[selection]

    return out


def collapse_index(array: np.ndarray, indices: np.ndarray):
    """

    Parameters
    ----------
    array: Array to select from.
        Must be reshaped such that the dimensions are indexed in the right order.
    indices: The indices to select.

    Returns
    -------
    out: The selected array.
    """

    indices = indices.astype(np.int).reshape(len(indices), -1)

    shape = array.shape[indices.shape[1] + 1:]

    out = np.empty((len(array), *shape), dtype=array.dtype)

    for i in range(len(indices)):
        index, a = indices[i], array[i]
        out_ = a
        for index_ in index:
            out_ = out_[index_]
        out[i] = out_

    return out


class SourdoughAgents(object):

    def __init__(self, n_initial: int = 10, n_max: int = 1000, size: int = 100,
                 mutation_rate: float = 1e-2, depletion_rate: float = .1, max_exchange_rate: float = .1):

        self.size_ = size

        self.directions_ = np.array([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ])

        self.position = np.random.choice(size, size=(n_max, 2))
        self.direction = np.random.choice(4, size=n_max)

        self.energy = np.ones(n_max) * .5 * (np.arange(n_max) < n_initial)
        self.q = np.random.randn(n_max, 2, 3)

        self.mutation_rate = mutation_rate
        self.depletion_rate = depletion_rate

        self.max_exchange_rate = max_exchange_rate

    def mitose(self):
        mitose = self.energy > 1
        n = mitose.sum()

        if n == 0:
            return

        dead = np.argsort(self.energy)[:n]

        self.energy[mitose] = .5
        self.energy[dead] = .5

        self.q[dead] = self.q[mitose]
        self.position[dead] = self.position[mitose]

        self.direction[dead] = np.random.choice(4, n)
        self.direction[mitose] = np.random.choice(4, n)

    def energise(self, energy):
        alive = self.energy > 1e-6

        energy = energy.reshape(-1)

        subset = self.position[alive]
        indices = subset[:, 0] * self.size_ + subset[:, 1]

        self.energy[alive] += energy.take(indices, axis=0)

        return self.energy[alive]

    def deplete(self):
        alive = self.energy > 1e-6
        self.energy[alive] -= self.depletion_rate

    def mutate(self):
        alive = self.energy > 1e-6
        self.q[alive] += np.random.randn(alive.sum(), 2, 3) + self.mutation_rate

    def move(self, energy):
        alive = self.energy > 1e-6

        if alive.sum() == 0:
            return

        energy = energy.reshape(-1)

        subset = self.position[alive]
        indices = subset[:, 0] * self.position.shape[1] + subset[:, 1]
        energy = (1 * (energy.take(indices, axis=0) > 0)).astype(np.int)
        actions = collapse_index(self.q[alive], energy).argmax(axis=1) - 1

        forward = actions == 0
        turn = actions != 0

        self.position[alive] += self.directions_[self.direction[alive]] * forward.reshape(-1, 1)
        self.direction[alive] += actions * turn

        self.position = self.position.clip(0, self.size_ - 1)
        self.direction = self.direction % 4

    def exchange_(self, indices):
        shuffle = np.random.choice(indices, replace=True, size=len(indices))
        self.q[indices] += np.random.uniform(0, self.max_exchange_rate) * (self.q[shuffle] - self.q[indices])

    def exchange(self):
        alive = self.energy > 1e-6

        indices = np.arange(len(self.position))[alive]
        position = self.position[indices]

        unique, counts = np.unique(position, return_counts=True)

        for u, c in zip(unique, counts):
            if c > 1:
                index = indices[(position == u).all(axis=1)]
                self.exchange_(index)

    def step(self, environment):

        energy = environment.step()
        self.energise(energy)
        self.mitose()
        self.deplete()
        self.mutate()
        self.exchange()
        self.move(energy)
