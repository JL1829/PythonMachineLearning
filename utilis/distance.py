import numpy as np


def euclidean(x, y):
    """
    Compute the Euclidean ("L2") distance between two vectors

    :param x: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :param y: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :return:  float, the L2 distance between 'x' vector and 'y' vector

    ----------
    Example:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([6, 7, 8, 9, 10])
    >>> euclidean(x, y)
    11.180339887498949
    """

    return np.sqrt(np.sum((x - y) ** 2))


def manhattan(x, y):
    """
    Compute the Manhattan ('L1') distance between two vectors

    :param x: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :param y: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :return:  float, the L1 distance between 'x' vector and 'y' vector

    ----------
    Example:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([6, 7, 8, 9, 10])
    >>> manhattan(x, y)
    25
    """
    return np.sum(np.abs(x - y))


def chebyshev(x, y):
    """
    Compute the Chebyshev ($$ L_\infty $$ ) distance between two vectors

    :param x: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :param y: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :return:  float, the Chebyshev distance between 'x' vector and 'y' vector

    ----------
    Example:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([6, 7, 8, 9, 10])
    >>> chebyshev(x, y)
    5
    """
    return np.max(np.abs(x - y))


def hamming(x, y):
    """
    Compute the Hamming distance between two integer-valued vectors

    :param x: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :param y: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :return:  float, the Hamming distance.

    ----------
    Example:
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([6, 7, 8, 9, 10])
    >>> hamming(x, y)
    1.0
    """
    return np.sum(x != y) / len(x)
