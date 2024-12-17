import numpy as np
from spatial.jcalc import jcalc
from spatial.get_gravity import get_gravity


def CMM(obj: 'ContinuousDynamics', sys: 'System') -> np.ndarray:
    """
    Calculate Centroidal Momentum Matrix.

    Parameters:
        obj (ContinuousDynamics): Object containing necessary methods
        sys (System): System containing model information

    Returns:
        np.ndarray: Centroidal Momentum Matrix (6 x nd)
    """
    model = sys.Model
    Xtree = model.Xtree

    q = sys.qpos0
    qd = sys.qvel0

    a_grav = get_gravity(sys.Model)

    idx1 = slice(0, 6)  # corresponds to free joint
    idx2 = [i + 6 for i in range(12)]  # corresponds to revolute joints
    v_idx = [idx1, *idx2]

    idx3 = [i + 7 for i in range(13)]  # corresponds to all joints
    q_idx = [slice(0, 7), *idx3]

    # Initialize composite inertia calculation
    Ic = model.fullinertia.copy()  # Make a copy to avoid modifying original
    I0 = np.zeros((6, 6))

    nd = model.params.nb
    parent_id = model.parent_id

    # Initialize CMM
    A = np.zeros((6, model.params.nv))

    # Initialize dictionaries
    Xup = {}
    S = {}
    XiG = {}

    # Backward pass for composite inertias
    for i in range(nd-1, 0, -1):
        XJ, S[i] = jcalc(model.jtype[i-1], q[q_idx[i-1]])
        Xup[i] = XJ @ Xtree[i]

        if parent_id[i] != 0:
            Ic[parent_id[i]] = Ic[parent_id[i]] + Xup[i].T @ Ic[i] @ Xup[i]
        else:
            I0 = I0 + Xup[i].T @ Ic[i] @ Xup[i]

    # Calculate centroidal transform
    M = I0[5, 5]  # For 3D case, use [5,5] instead of MATLAB's [6,6]
    pG = skew(I0[0:3, 3:6] / M)
    X0G = np.block([
        [np.eye(3), np.zeros((3, 3))],
        [skew(pG), np.eye(3)]
    ])

    # Forward pass to compute CMM
    for i in range(1, nd):
        if parent_id[i] != 0:
            XiG[i] = Xup[i] @ XiG[parent_id[i]]
        else:
            XiG[i] = Xup[i] @ X0G
        if i == 1:
            A[:, v_idx[i-1]] = (XiG[i].T @ Ic[i] @ S[i]).reshape((6, 6))
        else:
            A[:, v_idx[i - 1]] = (XiG[i].T @ Ic[i] @ S[i]).reshape((6,))

    return A


def skew(v: np.ndarray) -> np.ndarray:
    """
    Convert between 3D vector and 3x3 skew-symmetric matrix.

    Parameters:
        v (np.ndarray): Either 3D vector or 3x3 matrix

    Returns:
        np.ndarray: Either 3x3 skew-symmetric matrix or 3D vector
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
