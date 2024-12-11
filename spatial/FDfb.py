import numpy as np
from spatial.jcalc import jcalc
from spatial.crm import crm
from spatial.crf import crf
from spatial.rq import rq
from spatial.rqd import rqd
from spatial.plux import plux
from spatial.Vpt import Vpt
from spatial.get_gravity import get_gravity
from dynamics.apply_external_forces import apply_external_forces


def FDfb(model, xfb, q, qd, tau, f_ext=None):
    """
    Floating-Base Forward Dynamics via Articulated-Body Algorithm.

    Calculates forward dynamics of floating-base kinematic tree avoiding
    kinematic singularity in the six-joint chain.

    Parameters:
        model: Robot model structure
        xfb: 13-element state vector [quat_fb; pos_fb; vel_fb]
        q: Joint positions (real joints, 7 onwards)
        qd: Joint velocities
        tau: Joint forces
        f_ext: Optional external forces {nb_bodies} (first 5 ignored)

    Returns:
        xdfb: Time-derivative of xfb
        qdd: Joint accelerations
    """

    a_grav = get_gravity(model)

    # Extract floating base state
    qn = xfb[0:4]  # unit quaternion fixed-->f.b.
    r = xfb[4:7]   # position of f.b. origin
    Xup = {}
    Xup[6] = plux(rq(qn), r)  # xform fixed --> f.b. coords

    vfb = xfb[7:]
    v = {}
    v[6] = Xup[6] @ vfb  # f.b. vel in f.b. coords

    # Initialize articulated-body quantities
    IA = {}
    pA = {}
    c = {}
    U = {}
    d = {}
    u = {}
    a = {}
    S = {}

    IA[6] = model.I[6]
    pA[6] = crf(v[6]) @ model.I[6] @ v[6]

    # Forward pass
    for i in range(7, model.NB + 1):
        XJ, S[i] = jcalc(model.jtype[i], q[i-7])
        vJ = S[i] * qd[i-7]
        Xup[i] = XJ @ model.Xtree[i]
        v[i] = Xup[i] @ v[model.parent[i]] + vJ
        c[i] = crm(v[i]) @ vJ
        IA[i] = model.I[i]
        pA[i] = crf(v[i]) @ model.I[i] @ v[i]

    # Apply external forces if provided
    if f_ext is not None and len(f_ext) > 0:
        prnt = [x - 5 for x in model.parent[6:]]
        pA_list = list(pA.values())
        Xup_list = list(Xup.values())
        f_ext_list = f_ext[6:]
        pA_new = apply_external_forces(prnt, Xup_list, pA_list, f_ext_list)
        for i, p in enumerate(pA_new, 6):
            pA[i] = p

    # Backward pass
    for i in range(model.NB, 6, -1):
        U[i] = IA[i] @ S[i]
        d[i] = S[i].T @ U[i]
        u[i] = tau[i-7] - S[i].T @ pA[i]
        Ia = IA[i] - np.outer(U[i], U[i].T) / d[i]
        pa = pA[i] + Ia @ c[i] + U[i] * u[i] / d[i]
        IA[model.parent[i]] = IA[model.parent[i]] + Xup[i].T @ Ia @ Xup[i]
        pA[model.parent[i]] = pA[model.parent[i]] + Xup[i].T @ pa

    # Floating base acceleration without gravity
    a[6] = -np.linalg.solve(IA[6], pA[6])

    # Initialize qdd (avoids warning when NB==6)
    qdd = np.zeros(max(0, model.NB - 6))

    # Forward pass to compute accelerations
    for i in range(7, model.NB + 1):
        a[i] = Xup[i] @ a[model.parent[i]] + c[i]
        qdd[i-7] = (u[i] - U[i].T @ a[i]) / d[i]
        a[i] = a[i] + S[i] * qdd[i-7]

    qnd = rqd(vfb[0:3], qn)        # derivative of qn
    rd = Vpt(vfb, r)               # lin vel of flt base origin
    # true f.b. accn in fixed-base coords
    afb = np.linalg.solve(Xup[6], a[6]) + a_grav

    xdfb = np.concatenate([qnd, rd, afb])

    return xdfb, qdd
