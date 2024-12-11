import numpy as np


def skew(v):
    """Convert 3D vector to skew-symmetric matrix."""
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])


def skewCustom(mC):
    """Extract vector from skew-symmetric matrix * mass."""
    return np.array([mC[2, 1], mC[0, 2], mC[1, 0]])


def mcI(i1, i2=None, i3=None):
    """
    Convert between rigid-body inertia and mass, CoM and rotational inertia.

    Two usage modes:
    1. rbi = mcI(m, c, I): Convert mass, CoM, and inertia to rigid-body inertia matrix
    2. m, c, I = mcI(rbi): Convert rigid-body inertia matrix to mass, CoM, and inertia

    Parameters:
        i1: Either mass m or rigid-body inertia matrix rbi
        i2: Optional center of mass vector c
        i3: Optional rotational inertia I

    Returns:
        If converting to rbi: returns rigid-body inertia matrix
        If converting from rbi: returns mass, CoM vector, and rotational inertia
    """
    if i2 is None:
        return rbi_to_mcI(i1)
    else:
        return mcI_to_rbi(i1, i2, i3)


def mcI_to_rbi(m, c, I):
    """Convert mass, CoM and rotational inertia to rigid-body inertia matrix."""
    c = np.asarray(c).flatten()

    if len(c) == 3:  # spatial case
        C = skew(c)
        rbi = np.block([
            [I + m * C @ C.T, m * C],
            [m * C.T, m * np.eye(3)]
        ])
    else:  # planar case
        rbi = np.array([
            [I + m * np.dot(c, c), -m * c[1], m * c[0]],
            [-m * c[1], m, 0],
            [m * c[0], 0, m]
        ])

    return rbi


def rbi_to_mcI(rbi):
    """Convert rigid-body inertia matrix to mass, CoM and rotational inertia."""
    rbi = np.asarray(rbi)

    if rbi.shape == (6, 6):  # spatial case
        m = rbi[5, 5]
        mC = rbi[0:3, 3:6]
        c = skewCustom(mC) / m
        I = rbi[0:3, 0:3] - mC @ mC.T / m
        return m, c, I
    else:  # planar case
        m = rbi[2, 2]
        c = np.array([rbi[2, 0], -rbi[1, 0]]) / m
        I = rbi[0, 0] - m * np.dot(c, c)
        return m, c, I
