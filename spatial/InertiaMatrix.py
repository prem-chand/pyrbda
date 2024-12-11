import numpy as np


def InertiaMatrix(obj, sys, q, S, Xup):
    """
    Calculate the joint-space inertia matrix.

    Parameters:
        obj: Object containing necessary methods
        sys: System containing model information
        q: Joint position vector
        S: Dictionary of motion subspaces
        Xup: Dictionary of coordinate transforms

    Returns:
        H: Joint-space inertia matrix
    """

    model = sys.Model

    # Initialize composite inertia calculation
    IC = model.I.copy()  # Make a copy to avoid modifying original

    nd = model.nd
    parent = model.parent

    # Backward pass to compute composite inertias
    for i in range(nd, 0, -1):
        if parent[i] != 0:
            IC[parent[i]] = IC[parent[i]] + Xup[i].T @ IC[i] @ Xup[i]

    # Initialize inertia matrix
    H = np.zeros((nd, nd))

    # Forward pass to compute inertia matrix
    for i in range(1, nd + 1):
        fh = IC[i] @ S[i]
        H[i-1, i-1] = S[i].T @ fh

        j = i
        while parent[j] > 0:
            fh = Xup[j].T @ fh
            j = parent[j]
            H[i-1, j-1] = S[j].T @ fh
            H[j-1, i-1] = H[i-1, j-1]  # Symmetric matrix

    return H
