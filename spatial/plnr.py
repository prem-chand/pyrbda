import numpy as np


def plnr(i1, i2=None):
    """
    Compose/decompose planar-vector coordinate transform.

    Two usage modes:
    1. X = plnr(theta, r): Compose transform X from angle theta and vector r
    2. theta, r = plnr(X): Decompose transform X into angle theta and vector r

    Parameters:
        i1: Either rotation angle theta (in radians) or transform matrix X
        i2: Optional 2D position vector r (if composing transform)

    Returns:
        If composing (theta, r given): returns transform matrix X
        If decomposing (X given): returns theta, r
    """
    if i2 is not None:  # theta,r --> X
        theta = i1
        r = np.asarray(i2).flatten()  # Convert r to 1D array

        c = np.cos(theta)
        s = np.sin(theta)

        X = np.array([
            [1,          0,  0],
            [s*r[0]-c*r[1], c,  s],
            [c*r[0]+s*r[1], -s, c]
        ])

        return X

    else:  # X --> theta,r
        X = np.asarray(i1)

        c = X[1, 1]
        s = X[1, 2]

        theta = np.arctan2(s, c)
        r = np.array([
            [s*X[1, 0] + c*X[2, 0]],
            [s*X[2, 0] - c*X[1, 0]]
        ])

        return theta, r
