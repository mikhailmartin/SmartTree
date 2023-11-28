import functools

import numpy as np


def cat_partitions(collection: list) -> list[list]:
    """
    References:
        https://en.wikipedia.org/wiki/Partition_of_a_set
    """
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in cat_partitions(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def rank_partitions(collection: list) -> list[list]:
    for i in range(1, len(collection)):
        yield collection[:i], collection[i:]


def counter(function):
    """Декоратор-счётчик."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        wrapper.count += 1

        return function(*args, **kwargs)

    wrapper.count = 0

    return wrapper


def get_thresholds(array: np.ndarray) -> np.ndarray:
    array.sort()
    array = np.unique(array)
    thresholds = np.array([]) if len(array) == 1 else moving_average(array, 2)

    return thresholds


def moving_average(array: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(array, np.ones(window), mode='valid') / window
