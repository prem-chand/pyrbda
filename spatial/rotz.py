import numpy as np


def rotz(theta):
    """
    rotz  spatial coordinate transform (Z-axis rotation).
    rotz(theta) calculates the coordinate transform matrix from A to B
    coordinates for spatial motion vectors, where coordinate frame B is
    rotated by an angle theta (radians) relative to frame A about their
    common Z axis.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    X = np.array([[c,  s,  0,  0,  0,  0],
                  [-s, c,  0,  0,  0,  0],
                  [0,  0,  1,  0,  0,  0],
                  [0,  0,  0,  c,  s,  0],
                  [0,  0,  0, -s,  c,  0],
                  [0,  0,  0,  0,  0,  1]])
    return X
