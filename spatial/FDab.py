import numpy as np
from spatial.jcalc import jcalc
from spatial.crm import crm
from spatial.crf import crf
from spatial.get_gravity import get_gravity
from dynamics.apply_external_forces import apply_external_forces


def FDab(model, q, qd, tau, f_ext=None):
    """
    Forward Dynamics via Articulated-Body Algorithm.

    Calculates forward dynamics of a kinematic tree using the 
    articulated-body algorithm.

    Parameters:
        model: Robot model structure
        q: Joint position vector
        qd: Joint velocity vector
        tau: Joint force vector
        f_ext: Optional external forces

    Returns:
        qdd: Joint acceleration vector
    """

    a_grav = get_gravity(model)

    # Initialize dictionaries
    Xup = {}  # Coordinate transforms
    v = {}    # Spatial velocities
    c = {}    # Velocity products
    S = {}    # Motion subspaces
    IA = {}   # Articulated-body inertias
    pA = {}   # Articulated-body bias forces
    U = {}    # Intermediate calculations
    d = {}    # Intermediate calculations
    u = {}    # Intermediate calculations
    a = {}    # Spatial accelerations

    # Forward pass
    for i in range(1, model.NB + 1):
        XJ, S[i] = jcalc(model.jtype[i], q[i-1])
        vJ = S[i] * qd[i-1]
        Xup[i] = XJ @ model.Xtree[i]

        if model.parent[i] == 0:
            v[i] = vJ
            c[i] = np.zeros_like(a_grav)
        else:
            v[i] = Xup[i] @ v[model.parent[i]] + vJ
            c[i] = crm(v[i]) @ vJ

        IA[i] = model.I[i]
        pA[i] = crf(v[i]) @ model.I[i] @ v[i]

    # Apply external forces if provided
    if f_ext is not None:
        pA = apply_external_forces(model.parent, Xup, pA, f_ext)

    # Backward pass
    for i in range(model.NB, 0, -1):
        U[i] = IA[i] @ S[i]
        d[i] = S[i].T @ U[i]
        u[i] = tau[i-1] - S[i].T @ pA[i]

        if model.parent[i] != 0:
            Ia = IA[i] - np.outer(U[i], U[i].T) / d[i]
            pa = pA[i] + Ia @ c[i] + U[i] * u[i] / d[i]
            IA[model.parent[i]] = IA[model.parent[i]] + Xup[i].T @ Ia @ Xup[i]
            pA[model.parent[i]] = pA[model.parent[i]] + Xup[i].T @ pa

    # Forward pass to compute accelerations
    qdd = np.zeros(model.NB)

    for i in range(1, model.NB + 1):
        if model.parent[i] == 0:
            a[i] = Xup[i] @ (-a_grav) + c[i]
        else:
            a[i] = Xup[i] @ a[model.parent[i]] + c[i]

        qdd[i-1] = (u[i] - U[i].T @ a[i]) / d[i]
        a[i] = a[i] + S[i] * qdd[i-1]

    return qdd
