import numpy as np
from numba import njit

MAX_INT = 2 ** 32 - 1


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


@njit
def distance(position):
    n = len(position)
    d = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            d[i, j] = d[j, i] = ((position[i] - position[j]) ** 2).sum()

    return d
