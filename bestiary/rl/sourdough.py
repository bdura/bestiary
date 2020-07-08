from typing import Optional

import numpy as np
from loguru import logger
from numba import njit
from sklearn.neighbors import KDTree
from tqdm import tqdm


@njit
def select_(array, indices):
    out = np.empty((len(indices), *array.shape[1:]))
    for i in range(len(indices)):
        out[i] = array[indices[i]]
    return out


def select(array, indices):
    indices = indices.reshape(len(indices), -1).astype(int)

    a = select_(array, indices[:, 0])

    for i in indices.T[1:]:
        a = select_along_(a, i)

    return a


@njit
def select_along_(array, indices):
    out = np.empty((len(indices), *array.shape[2:]))
    for i in range(len(indices)):
        out[i] = array[i, indices[i]]
    return out


def select_along(array, indices):
    indices = indices.reshape(len(indices), -1).astype(int)

    a = array
    for i in indices.T:
        a = select_along_(a, i)

    return a


@njit
def stratified_index_(index, sub_index, out):
    si = 0
    for i in range(len(index)):

        if index[i]:
            out[i] = sub_index[si]
            si += 1
        else:
            out[i] = False

    return out


def stratified_index(index, sub_index):
    out = np.empty((len(index),), dtype=bool)
    return stratified_index_(index, sub_index, out)


class FoodSource(object):

    def __init__(self, size: int = 100, energy: float = 1, bandwidth: float = 2., speed: float = 100.):
        self.size_ = size
        self.energy = energy
        self.bandwidth = bandwidth

        self.alpha = np.random.uniform(0, np.pi)
        self.rho = .8 * size

        self.omega = speed / self.rho

    def position(self):
        return np.array([np.cos(self.alpha), np.sin(self.alpha)]) * self.rho

    def move(self):
        self.alpha += np.random.uniform(self.omega / 2, self.omega)

    def food(self, position, center=None):
        if center is None:
            center = self.position()
        center = center.reshape(1, -1)
        return self.energy * np.exp(- (np.linalg.norm(position - center, axis=1) / self.bandwidth) ** 2)


class Ecoli(object):

    def __init__(self, size: int = 100, n_max: int = 1000, energy_levels: int = 2,
                 max_food: float = .5, food_levels: int = 3, depletion_rate: float = .05,
                 mutation_rate: float = .1, bandwidth: float = 5., speed: float = .5,
                 record: bool = False, seed: Optional[int] = None):

        np.random.seed(seed)

        self.size_ = size

        self.depletion_rate = depletion_rate
        self.mutation_rate = mutation_rate

        self.position = np.random.uniform(-size, size, size=(n_max, 2))
        self.direction = np.random.uniform(0, 2 * np.pi, size=n_max)

        self.energy = np.ones(n_max) * .5

        # s = (E, F, dF)
        self.q = np.random.randn(n_max, energy_levels, food_levels, 3, 2)

        self.energy_levels_ = np.linspace(0, .9, energy_levels)
        self.food_levels_ = np.linspace(0, max_food, food_levels)

        self.level_ = np.zeros(n_max)

        self.alive = self.energy > 0

        self.food_source = FoodSource(size=size, energy=max_food * 1.1, bandwidth=bandwidth, speed=speed)

        self.history_ = []
        self.record = record

    def run(self, indices):

        if indices.sum() == 0:
            return

        alpha = self.direction[indices]
        u = np.array([np.cos(alpha), np.sin(alpha)]).T

        self.position[indices] += u * np.random.normal(1., 1e-1, size=len(u)).reshape(-1, 1)

    def tumble(self, indices):

        if indices.sum() == 0:
            return

        self.direction[indices] = np.random.uniform(0, 2 * np.pi, size=indices.sum())

    def deplete(self):
        self.energy -= self.depletion_rate
        self.alive = self.energy > 0

        self.level_[~self.alive] = 0

    def mutate(self):
        self.q[self.alive] += np.random.normal(0, self.mutation_rate, size=self.q[self.alive].shape)

    def mitose(self):

        mitose = self.energy > 1
        dead = np.argsort(self.energy)[:mitose.sum()]
        dead = np.isin(np.arange(len(self.position)), dead)

        self.energy[mitose] /= 2
        self.energy[dead] = self.energy[mitose]

        self.q[dead] = self.q[mitose]

        self.position[dead] = self.position[mitose]
        self.direction[dead] = np.random.uniform(0, 2 * np.pi, size=dead.sum())

        self.alive = np.logical_or(self.alive, dead)

        n = mitose.sum()
        if n > 0:
            logger.info(f"{n} bacteria just mitosed")

    def discretise(self, energy, food):

        e = np.digitize(energy, self.energy_levels_) - 1
        f = np.digitize(food, self.food_levels_) - 1

        df = np.clip(f - self.level_[self.alive], -1, 1) + 1
        self.level_[self.alive] = f

        return e, f, df

    def act(self, food):
        energy = self.energy[self.alive]
        q = self.q[self.alive]

        e, f, df = self.discretise(energy, food)

        q = select_along(q, e)
        q = select_along(q, f)
        q = select_along(q, df)

        actions = q.argmax(axis=1)

        run = stratified_index(self.alive, actions == 0)
        tumble = stratified_index(self.alive, actions == 1)

        self.run(run)
        self.tumble(tumble)

    def step(self):

        self.food_source.move()
        food = self.food_source.food(self.position)

        self.energy[self.alive] += food[self.alive]
        self.mitose()
        self.deplete()

        if self.alive.sum() > 0:
            self.mutate()
            self.act(food[self.alive])

        if self.record:
            record = dict(
                alive=int(self.alive.sum()),
                energy=self.energy[self.alive],
                q=self.q[self.alive],
                position=self.position[self.alive],
                direction=self.direction[self.alive],
                food=self.food_source.position(),
            )

            self.history_.append(record)

    def conjugate(self):

        position = self.position[self.alive]
        q = self.q[self.alive]

        tree = KDTree(position)

        d, i = tree.query(position, k=2)
        distance, nn = d[:, 1], i[:, 1]

        conjugate = distance < 1.
        conjugate_from = nn[conjugate]

        q[conjugate] += np.random.rand(q[conjugate].shape) * .1 * (q[conjugate_from] - q[conjugate])

    def simulate(self, generations=1000, verbose=False):
        iterator = range(generations)

        if verbose:
            iterator = tqdm(iterator, ascii=True, ncols=100)

        for _ in iterator:
            self.step()

            if self.alive.sum() == 0:
                break
