import numpy as np


def CoriolisAndGravityTerms(sys, q, S, Xup, fvp):
    """
    Calculate Coriolis, centrifugal, and gravity terms.

    Parameters:
        sys: System containing model information
        q: Joint position vector
        S: Dictionary of motion subspaces
        Xup: Dictionary of coordinate transforms
        fvp: Dictionary of velocity-product forces

    Returns:
        C: Vector of Coriolis, centrifugal, and gravity terms
    """

    model = sys.Model
    parent = model.parent
    nd = model.params.nj  # number of joints

    # Initialize C vector
    if isinstance(q, np.ndarray):
        C = np.zeros((model.params.nv, 1))
    else:
        C = 0*sys.States['dq']

    idx1 = slice(0, 6)  # corresponds to free joint
    idx2 = [i+6 for i in range(12)]  # corresponds to revolute joints
    v_idx = [idx1, *idx2]

    # Backward pass to accumulate forces
    for i in range(nd-1, 0, -1):
        C[v_idx[i]] = S[i].T @ fvp[i]
        fvp[model.parent_id[i]] = fvp[model.parent_id[i]] + \
                Xup[i].T @ fvp[i]

    C[:6] = S[0].T @ fvp[0]

    return C
