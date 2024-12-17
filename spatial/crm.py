import numpy as np
import casadi as ca


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
    if isinstance(v, np.ndarray):
        v = v.flatten().copy()
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
    else:
        if v.shape[0] == 6:
            vcross = ca.SX.zeros(6, 6)
            vcross[0, 1] = -v[2]
            vcross[0, 2] = v[1]
            vcross[1, 0] = v[2]
            vcross[1, 2] = -v[0]
            vcross[2, 0] = -v[1]
            vcross[2, 1] = v[0]
            vcross[3, 1] = -v[5]
            vcross[3, 2] = v[4]
            vcross[3, 4] = -v[2]
            vcross[3, 5] = v[1]
            vcross[4, 0] = v[5]
            vcross[4, 2] = -v[3]
            vcross[4, 3] = v[2]
            vcross[4, 5] = -v[0]
            vcross[5, 0] = -v[4]
            vcross[5, 1] = v[3]
            vcross[5, 3] = -v[1]
            vcross[5, 4] = v[0]
        else:
            vcross = ca.SX.zeros(3, 3)
            vcross[0, 1] = -v[2]
            vcross[0, 2] = v[1]
            vcross[1, 0] = v[2]
            vcross[1, 2] = -v[0]
            vcross[2, 0] = -v[1]
            vcross[2, 1] = v[0]

    return vcross


# import numpy as np


# def crm(v):
#     """
#     Spatial/planar cross-product operator (motion).

#     Calculates the 6x6 (or 3x3) matrix such that crm(v)*m is the cross product
#     of the motion vectors v and m.

#     Parameters:
#         v: Motion vector (6D for spatial, 3D for planar)

#     Returns:
#         vcross: Cross-product operator matrix (6x6 for spatial, 3x3 for planar)
#     """

#     if len(v) == 6:  # spatial vector
#         vcross = np.array([
#             [0,    -v[2],  v[1],   0,     0,     0],
#             [v[2],  0,    -v[0],   0,     0,     0],
#             [-v[1], v[0],  0,      0,     0,     0],
#             [0,    -v[5],  v[4],   0,    -v[2],  v[1]],
#             [v[5],  0,    -v[3],   v[2],  0,    -v[0]],
#             [-v[4], v[3],  0,     -v[1],  v[0],  0]
#         ])
#     else:  # planar vector
#         vcross = np.array([
#             [0,     0,     0],
#             [v[2],  0,    -v[0]],
#             [-v[1], v[0],  0]
#         ])

#     return vcross
