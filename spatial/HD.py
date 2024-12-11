import numpy as np
from spatial.jcalc import jcalc
from spatial.crm import crm
from spatial.crf import crf
from spatial.get_gravity import get_gravity
from dynamics.apply_external_forces import apply_external_forces


def HD(model, fd, q, qd, qdd, tau, f_ext=None):
    """
    Articulated-Body Hybrid Dynamics Algorithm.

    Calculates hybrid dynamics of a kinematic tree using the articulated-body algorithm.

    Parameters:
        model: Robot model structure
        fd: Boolean array (fd[i]=True for forward-dynamics joints)
        q: Joint positions
        qd: Joint velocities
        qdd: Joint accelerations (used when fd[i]=False)
        tau: Joint forces (used when fd[i]=True)
        f_ext: Optional external forces

    Returns:
        qdd_out: Joint accelerations (calculated where fd[i]=True)
        tau_out: Joint forces (calculated where fd[i]=False)
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

        if not fd[i-1]:  # Inverse dynamics joint
            c[i] = c[i] + S[i] * qdd[i-1]

        IA[i] = model.I[i]
        pA[i] = crf(v[i]) @ model.I[i] @ v[i]

    # Apply external forces if provided
    if f_ext is not None:
        pA = apply_external_forces(model.parent, Xup, pA, f_ext)

    # Backward pass
    for i in range(model.NB, 0, -1):
        if not fd[i-1]:  # Inverse dynamics joint
            if model.parent[i] != 0:
                Ia = IA[i]
                pa = pA[i] + IA[i] @ c[i]
                IA[model.parent[i]] = IA[model.parent[i]] + \
                    Xup[i].T @ Ia @ Xup[i]
                pA[model.parent[i]] = pA[model.parent[i]] + Xup[i].T @ pa
        else:  # Forward dynamics joint
            U[i] = IA[i] @ S[i]
            d[i] = S[i].T @ U[i]
            u[i] = tau[i-1] - S[i].T @ pA[i]
            if model.parent[i] != 0:
                Ia = IA[i] - np.outer(U[i], U[i].T) / d[i]
                pa = pA[i] + Ia @ c[i] + U[i] * u[i] / d[i]
                IA[model.parent[i]] = IA[model.parent[i]] + \
                    Xup[i].T @ Ia @ Xup[i]
                pA[model.parent[i]] = pA[model.parent[i]] + Xup[i].T @ pa

    # Forward pass to compute accelerations and forces
    qdd_out = np.zeros(model.NB)
    tau_out = np.zeros(model.NB)

    for i in range(1, model.NB + 1):
        if model.parent[i] == 0:
            a[i] = Xup[i] @ (-a_grav) + c[i]
        else:
            a[i] = Xup[i] @ a[model.parent[i]] + c[i]

        if not fd[i-1]:  # Inverse dynamics joint
            qdd_out[i-1] = qdd[i-1]
            tau_out[i-1] = S[i].T @ (IA[i] @ a[i] + pA[i])
        else:  # Forward dynamics joint
            qdd_out[i-1] = (u[i] - U[i].T @ a[i]) / d[i]
            tau_out[i-1] = tau[i-1]
            a[i] = a[i] + S[i] * qdd_out[i-1]

    return qdd_out, tau_out
