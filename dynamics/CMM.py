import numpy as np
from spatial.jcalc import jcalc
from spatial.get_gravity import get_gravity


def CMM(obj, sys):
    """
    Calculate Centroidal Momentum Matrix.

    Parameters:
        obj: Object containing necessary methods
        sys: System containing model information

    Returns:
        A: Centroidal Momentum Matrix (6 x nd)
    """

    model = sys.Model
    Xtree = model.Xtree

    q = sys.States.q.sym
    qd = sys.States.dq.sym

    a_grav = get_gravity(sys.Model)

    # Initialize composite inertia calculation
    IC = model.I.copy()  # Make a copy to avoid modifying original
    I0 = np.zeros((6, 6))

    nd = model.nd
    parent = model.parent

    # Initialize CMM
    A = np.zeros((6, nd))

    # Initialize dictionaries
    Xup = {}
    S = {}
    XiG = {}

    # Backward pass for composite inertias
    for i in range(nd, 0, -1):
        XJ, S[i] = jcalc(model.jtype[i], q[i-1])
        Xup[i] = XJ @ Xtree[i]

        if parent[i] != 0:
            IC[parent[i]] = IC[parent[i]] + Xup[i].T @ IC[i] @ Xup[i]
        else:
            I0 = I0 + Xup[i].T @ IC[i] @ Xup[i]

    # Calculate centroidal transform
    M = I0[5, 5]  # For 3D case, use [5,5] instead of MATLAB's [6,6]
    pG = skew(I0[0:3, 3:6] / M)
    X0G = np.block([
        [np.eye(3), np.zeros((3, 3))],
        [skew(pG), np.eye(3)]
    ])

    # Forward pass to compute CMM
    for i in range(1, nd + 1):
        if parent[i] != 0:
            XiG[i] = Xup[i] @ XiG[parent[i]]
        else:
            XiG[i] = Xup[i] @ X0G

        A[:, i-1] = XiG[i].T @ IC[i] @ S[i]

    return A


def skew(v):
    """
    Convert between 3D vector and 3x3 skew-symmetric matrix.

    Parameters:
        v: Either 3D vector or 3x3 matrix

    Returns:
        Either 3x3 skew-symmetric matrix or 3D vector
    """

    if isinstance(v, np.ndarray) and v.shape == (3, 3):
        # Convert matrix to vector
        return 0.5 * np.array([
            v[2, 1] - v[1, 2],
            v[0, 2] - v[2, 0],
            v[1, 0] - v[0, 1]
        ])
    else:
        # Convert vector to matrix
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
