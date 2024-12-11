import numpy as np


def Vpt(v, p):
    """
    Calculate linear velocities at points from spatial/planar velocities.

    Parameters:
        v: Spatial/planar velocities (6×1 or 6×n for 3D, 3×1 or 3×n for 2D)
        p: Points (3×n for 3D, 2×n for 2D)

    Returns:
        vp: Linear velocities at points (3×n for 3D, 2×n for 2D)

    Note: If v is a single vector, it applies to all points in p.
          Otherwise, vp[:,i] is calculated from v[:,i] and p[:,i].
    """

    # Replicate v if single velocity vector with multiple points
    if v.shape[1] == 1 and p.shape[1] > 1:
        v = np.tile(v, (1, p.shape[1]))

    if v.shape[0] == 6:  # 3D points and velocities
        vp = v[3:6, :] + np.cross(v[0:3, :].T, p.T).T
    else:  # 2D points and velocities
        vp = np.vstack([
            v[1:3, :] + np.array([-v[0, :] * p[1, :],
                                 v[0, :] * p[0, :]])
        ])

    return vp
