import numpy as np
from spatial.jcalc import jcalc


def He(model, q):
    """
    Calculate joint-space inertia matrix.

    Calculates the joint-space inertia matrix H(q) from the equation of motion:
    tau = H(q)qdd + C(q,qd,f_ext)

    Parameters:
        model: Robot model structure
        q: Joint position vector

    Returns:
        He: Joint-space inertia matrix
    """

    # Initialize dictionaries
    S = {}      # Motion subspaces
    Xup = {}    # Coordinate transforms

    # Forward pass: calculate coordinate transforms
    for i in range(1, model.NB + 1):
        XJ, S[i] = jcalc(model.jtype[i], q[i-1])
        Xup[i] = XJ @ model.Xtree[i]

    # Initialize composite inertia calculation
    IC = model.I.copy()  # Make a copy to avoid modifying the model

    # Backward pass: accumulate composite inertias
    for i in range(model.NB, 0, -1):
        if model.parent[i] != 0:
            IC[model.parent[i]] = IC[model.parent[i]] + \
                Xup[i].T @ IC[i] @ Xup[i]

    # Calculate joint-space inertia matrix
    He = np.zeros((model.NB, model.NB))

    for i in range(1, model.NB + 1):
        fh = IC[i] @ S[i]

        He[i-1, i-1] = S[i].T @ fh
        j = i
        while model.parent[j] > 0:
            fh = Xup[j].T @ fh
            j = model.parent[j]
            He[i-1, j-1] = S[j].T @ fh
            He[j-1, i-1] = He[i-1, j-1]  # Symmetric matrix

    return He
