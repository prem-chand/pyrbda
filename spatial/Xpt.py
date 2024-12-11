import numpy as np
from spatial.skew import skew


def Xpt(X, p):
    """
    Apply Plucker/planar coordinate transform to 2D/3D points.

    Parameters:
        X: Coordinate transform matrix (6x6 for Plucker, 3x3 for planar)
        p: Points matrix (3×n for 3D points, 2×n for 2D points)

    Returns:
        xp: Transformed points in new coordinates
    """

    if X.shape == (6, 6):  # 3D points
        E = X[0:3, 0:3]
        r = -skew(E.T @ X[3:6, 0:3])
    else:  # 2D points
        E = X[1:3, 1:3]
        r = np.array([
            X[1, 2]*X[1, 0] + X[2, 2]*X[2, 0],
            X[1, 2]*X[2, 0] - X[2, 2]*X[1, 0]
        ])

    # Replicate r if multiple points
    if p.shape[1] > 1:
        r = np.tile(r.reshape(-1, 1), (1, p.shape[1]))

    xp = E @ (p - r)
    return xp
