import numpy as np


def plux(i1, i2=None):
    """
    plux  compose/decompose Plucker coordinate transform.
    X=plux(E,r) and [E,r]=plux(X) compose a Plucker coordinate transform X
    from its component parts E and r, and decompose it into those parts,
    respectively. E is a 3x3 rotational coordinate transform and r is a 3D
    vector. r is returned as a column vector, but it can be supplied as a
    row or column vector. X is a coordinate transform corresponding to a
    shift of origin by an amount specified by r, followed by a rotation about
    the new origin as specified by E. For example, plux(rx(1),[2 3 4]) makes
    the same transform as rotx(1)*xlt([2 3 4]). If two arguments are
    supplied then they are assumed to be E and r, otherwise X.
    """
    if i2 is not None:  # E, r --> X
        E = i1
        r = np.asarray(i2).flatten()
        X = np.block([
            [E, np.zeros((3, 3))],
            [-E @ skew(r), E]
        ])
        return X
    else:  # X --> E, r
        X = i1
        E = X[:3, :3]
        r = -skew(E.T @ X[3:6, :3])
        return E, r


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
