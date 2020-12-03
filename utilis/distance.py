import numpy as np


def euclidean(x, y):
    """
    Compute the Euclidean ("L2") distance between two vectors

    :param x: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :param y: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :return:  float, the L2 distance between 'x' vector and 'y' vector
    """
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan(x, y):
    """
    Compute the Manhattan ('L1') distance between two vectors

    :param x: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :param y: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :return:  float, the L1 distance between 'x' vector and 'y' vector
    """
    return np.sum(np.abs(x - y))


def chebyshev(x, y):
    """
    Compute the Chebyshev ($$ L_\infty $$ ) distance between two vectors

    :param x: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :param y: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :return:  float, the Chebyshev distance between 'x' vector and 'y' vector
    """
    return np.max(np.abs(x - y))


def hamming(x, y):
    """
    Compute the Hamming distance between two integer-valued vectors

    :param x: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :param y: {array-like} 'ndarray <numpy ndarray>' of shape: [vector_length, ]
    :return:  float, the Hamming distance.
    """
    return np.sum(x != y) / len(x)
