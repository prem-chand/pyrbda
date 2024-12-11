import numpy as np


def CoriolisAndGravityTerms(obj, sys, q, S, Xup, fvp):
    """
    Calculate Coriolis, centrifugal, and gravity terms.

    Parameters:
        obj: Object containing necessary methods
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
    nd = model.nd

    # Initialize C vector
    C = np.zeros(nd)

    # Backward pass to accumulate forces
    for i in range(nd, 0, -1):
        C[i-1] = S[i].T @ fvp[i]

        if parent[i] != 0:
            fvp[parent[i]] = fvp[parent[i]] + Xup[i].T @ fvp[i]

    return C
