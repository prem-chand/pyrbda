import numpy as np


def crm(v):
    """
    Spatial/planar cross-product operator (motion).

    Calculates the 6x6 (or 3x3) matrix such that crm(v)*m is the cross product
    of the motion vectors v and m.

    Parameters:
        v: Motion vector (6D for spatial, 3D for planar)

    Returns:
        vcross: Cross-product operator matrix (6x6 for spatial, 3x3 for planar)
    """

    if len(v) == 6:  # spatial vector
        vcross = np.array([
            [0,    -v[2],  v[1],   0,     0,     0],
            [v[2],  0,    -v[0],   0,     0,     0],
            [-v[1], v[0],  0,      0,     0,     0],
            [0,    -v[5],  v[4],   0,    -v[2],  v[1]],
            [v[5],  0,    -v[3],   v[2],  0,    -v[0]],
            [-v[4], v[3],  0,     -v[1],  v[0],  0]
        ])
    else:  # planar vector
        vcross = np.array([
            [0,     0,     0],
            [v[2],  0,    -v[0]],
            [-v[1], v[0],  0]
        ])

    return vcross
