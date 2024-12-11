import numpy as np
from spatial.jcalc import jcalc
from spatial.crm import crm
from spatial.crf import crf
from spatial.get_gravity import get_gravity
from dynamics.apply_external_forces import apply_external_forces


def ID(model, q, qd, qdd, f_ext=None):
    """
    Inverse Dynamics via Recursive Newton-Euler Algorithm.

    Calculates the inverse dynamics of a kinematic tree using the recursive
    Newton-Euler algorithm.

    Parameters:
        model: Robot model structure
        q: Joint positions
        qd: Joint velocities
        qdd: Joint accelerations
        f_ext: Optional external forces acting on the bodies

    Returns:
        tau: Joint forces required to achieve the motion
    """

    a_grav = get_gravity(model)

    # Initialize dictionaries for variables
    Xup = {}  # Coordinate transforms
    v = {}    # Spatial velocities
    a = {}    # Spatial accelerations
    f = {}    # Spatial forces
    S = {}    # Motion subspaces

    # Forward pass
    for i in range(1, model.NB + 1):
        XJ, S[i] = jcalc(model.jtype[i], q[i-1])
        vJ = S[i] * qd[i-1]
        Xup[i] = XJ @ model.Xtree[i]

        if model.parent[i] == 0:
            v[i] = vJ
            a[i] = Xup[i] @ (-a_grav) + S[i] * qdd[i-1]
        else:
            v[i] = Xup[i] @ v[model.parent[i]] + vJ
            a[i] = Xup[i] @ a[model.parent[i]] + \
                S[i] * qdd[i-1] + crm(v[i]) @ vJ

        f[i] = model.I[i] @ a[i] + crf(v[i]) @ model.I[i] @ v[i]

    # Apply external forces if provided
    if f_ext is not None:
        f = apply_external_forces(model.parent, Xup, f, f_ext)

    # Backward pass
    tau = np.zeros(model.NB)
    for i in range(model.NB, 0, -1):
        tau[i-1] = S[i].T @ f[i]
        if model.parent[i] != 0:
            f[model.parent[i]] = f[model.parent[i]] + Xup[i].T @ f[i]

    return tau
