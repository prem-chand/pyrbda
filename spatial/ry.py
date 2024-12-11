import numpy as np


def ry(theta):
    """
    ry  3x3 coordinate rotation (Y-axis)
    ry(theta) calculates the 3x3 rotational coordinate transform matrix from
    A to B coordinates, where coordinate frame B is rotated by an angle theta
    (radians) relative to frame A about their common Y axis.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    E = np.array([[c, 0, -s],
                  [0, 1, 0],
                  [s, 0, c]])
    return E
