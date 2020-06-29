from typing import Union

import numpy as np


def collapse(angle: Union[float, np.ndarray]):
    """
    Collapses an angle to the ]-pi, pi] interval

    Parameters
    ----------
    angle: Angle to collapse

    Returns
    -------
    angle: Collapsed angle.

    Examples
    --------
    >>> collapse(3 * np.pi / 2) / np.pi
    0.5
    """
    angle %= (2 * np.pi)
    angle -= (angle > np.pi) * np.pi * 2
    return angle


def unit(alpha):
    return np.stack((np.cos(alpha), np.sin(alpha))).T


class Sourdough(object):

    def __init__(self, energy_mu: float = 1, energy_bw: float = 5.,
                 n_initial: int = 1 - 00, n_max: int = 100000, stride: float = 1.,
                 mutation_rate: float = 1e-1, depletion_rate: float = .1, max_exchange_rate: float = .1,
                 beacon_stride: float = 1., alpha_var: float = .1, alpha_mu: float = .05):
        self.beacon_position = np.random.uniform(-10, 10, size=2)
        self.beacon_alpha = np.random.uniform(- np.pi, np.pi)
        self.beacon_stride = beacon_stride
        self.alpha_mu = alpha_mu
        self.alpha_var = alpha_var

        self.energy_mu = energy_mu
        self.energy_bw = energy_bw

        self.position = np.random.uniform(-10, 10, size=(n_max, 2))
        self.direction = np.random.uniform(- np.pi, np.pi, size=n_max)

        self.energy = np.ones(n_max) * .5 * (np.arange(n_max) < n_initial)

        # Actions: \alpha < - \pi / 4, \alpha \in [-\pi / 4, \pi / 4], \alpha > \pi / 4
        self.q = np.random.randn(n_max, 3, 3)  # N x D x A

        self.stride = stride

        self.mutation_rate = mutation_rate
        self.depletion_rate = depletion_rate

        self.max_exchange_rate = max_exchange_rate

    @property
    def alive(self):
        return self.energy > 1e-6

    def move_beacon(self):
        self.beacon_alpha += np.random.normal(self.alpha_mu, scale=self.alpha_var)
        self.beacon_alpha = collapse(self.beacon_alpha)
        e = np.array([np.cos(self.beacon_alpha), np.sin(self.beacon_alpha)])
        self.beacon_position = self.beacon_position + np.random.uniform(0, self.beacon_stride) * e

    def bacteria_angle(self):
        position = self.position[self.alive]
        direction = self.direction[self.alive]

        x, y = (self.beacon_position - position).T
        np.arctan2(y, x)

        alpha = collapse(np.arctan2(y, x) - direction)

        return alpha

    def bacteria_quadrant(self):
        angle = self.bacteria_angle()
        action = -1 * (angle < - np.pi / 4) + 1 * (angle > np.pi / 4) + 1
        return action

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

        self.direction[dead] = np.random.uniform(0, 2 * np.pi)
        self.direction[mitose] = np.random.uniform(0, 2 * np.pi)

    def energise(self):
        position = self.position[self.alive]
        distance = np.linalg.norm(position - self.beacon_position, axis=1)

        energy = self.energy_mu * np.exp(- distance ** 2 / self.energy_bw)

        self.energy[self.alive] += energy

    def deplete(self):
        self.energy[self.alive] -= self.depletion_rate

    def mutate(self):
        self.q[self.alive] += np.random.normal(0, self.mutation_rate, size=(self.alive.sum(), 3, 3))

    def move(self):
        quadrant = self.bacteria_quadrant()
        q = collapse_index(self.q[self.alive], quadrant)
        action = q.argmax(axis=1) - 1

        self.direction[self.alive] += np.random.uniform(0, np.pi) * action

        e = unit(self.direction[self.alive])
        self.position[self.alive] += e * np.random.uniform(0, self.stride)

    def step(self):
        self.move_beacon()
        self.energise()
        self.mitose()
        self.mutate()
        self.move()
        self.deplete()

    def test(self):
        self.move_beacon()
        self.move()


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
