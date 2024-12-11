import numpy as np


def rx(theta):
    """
    rx  3x3 coordinate rotation (X-axis)
    rx(theta) calculates the 3x3 rotational coordinate transform matrix from
    A to B coordinates, where coordinate frame B is rotated by an angle theta
    (radians) relative to frame A about their common X axis.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    E = np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])
    return E
