import numpy as np


def XtoV(X):
    """
    Obtain spatial/planar vector from small-angle transform.

    Interprets X as the coordinate transform from A to B coordinates, which
    implicitly defines the location of frame B relative to frame A. Calculates
    the velocity of a third frame, C(t), that travels at constant velocity
    from frame A to B in one time unit.

    Parameters:
        X: Coordinate transform matrix (6x6 for spatial, 3x3 for planar)

    Returns:
        v: Velocity vector (6D for spatial, 3D for planar)
           This is exact only if frames A and B are parallel

    Note: The return value is an invariant of X (i.e., v=X@v), and can be
          regarded as being expressed in both A and B coordinates.
    """

    if X.shape == (6, 6):  # Plucker transform -> spatial vector
        v = 0.5 * np.array([
            X[1, 2] - X[2, 1],
            X[2, 0] - X[0, 2],
            X[0, 1] - X[1, 0],
            X[4, 2] - X[5, 1],
            X[5, 0] - X[3, 2],
            X[3, 1] - X[4, 0]
        ])

    else:  # planar transform -> planar vector
        v = np.array([
            X[1, 2],
            (X[2, 0] + X[1, 1]*X[2, 0] + X[1, 2]*X[1, 0])/2,
            (-X[1, 0] - X[1, 1]*X[1, 0] + X[1, 2]*X[2, 0])/2
        ])

    return v
