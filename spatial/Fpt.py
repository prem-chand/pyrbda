import numpy as np


def Fpt(fp, p):
    """
    Convert forces at points to spatial/planar forces.

    Parameters:
        fp: Linear forces (3×n for 3D, 2×n for 2D)
        p: Points where forces act (3×n for 3D, 2×n for 2D)

    Returns:
        f: Equivalent spatial/planar forces (6×n for 3D, 3×n for 2D)
           where f[:,i] is equivalent to fp[:,i] acting at p[:,i]
    """

    if fp.shape[0] == 3:  # 3D forces at 3D points
        f = np.vstack([
            np.cross(p.T, fp.T).T,
            fp
        ])
    else:  # 2D forces at 2D points
        f = np.vstack([
            p[0, :] * fp[1, :] - p[1, :] * fp[0, :],
            fp
        ])

    return f
