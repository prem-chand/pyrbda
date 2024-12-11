import numpy as np


def rz(theta):
    """
    rz  3x3 coordinate rotation (Z-axis)
    rz(theta) calculates the 3x3 rotational coordinate transform matrix from
    A to B coordinates, where coordinate frame B is rotated by an angle theta
    (radians) relative to frame A about their common Z axis.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    E = np.array([[c, s, 0],
                  [-s, c, 0],
                  [0, 0, 1]])
    return E
