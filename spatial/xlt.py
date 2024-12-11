import numpy as np


def xlt(r):
    """
    xlt  spatial coordinate transform (translation of origin).
    xlt(r) calculates the coordinate transform matrix from A to B
    coordinates for spatial motion vectors, in which frame B is translated by
    an amount r (3D vector) relative to frame A. r can be a row or column vector.
    """
    X = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, r[2], -r[1], 1, 0, 0],
        [-r[2], 0, r[0], 0, 1, 0],
        [r[1], -r[0], 0, 0, 0, 1]
    ])
    return X
