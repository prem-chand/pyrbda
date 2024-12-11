import numpy as np
from spatial.rq import rq
from spatial.rqd import rqd
from spatial.plux import plux
from spatial.jcalc import jcalc
from spatial.crm import crm
from spatial.crf import crf
from spatial.Vpt import Vpt
from spatial.get_gravity import get_gravity
from dynamics.apply_external_forces import apply_external_forces


def IDfb(model, xfb, q, qd, qdd, f_ext=None):
    """
    Floating-Base Inverse Dynamics (=Hybrid Dynamics).

    Calculates inverse dynamics of floating-base kinematic tree using 
    singularity-free representation (Table 9.6 RBDA).

    Parameters:
        model: Robot model structure
        xfb: 13-element state vector [quat_fb; pos_fb; vel_fb]
        q: Joint positions (real joints, 7 onwards)
        qd: Joint velocities
        qdd: Joint accelerations
        f_ext: Optional external forces {nb_bodies} (first 5 ignored)

    Returns:
        xdfb: Time-derivative of xfb
        tau: Joint forces to achieve qdd
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

    a = {}
    a[6] = np.zeros(6)

    IC = {}
    pC = {}
    IC[6] = model.I[6]
    pC[6] = model.I[6] @ a[6] + crf(v[6]) @ model.I[6] @ v[6]

    S = {}
    for i in range(7, model.NB + 1):
        XJ, S[i] = jcalc(model.jtype[i], q[i-7])
        vJ = S[i] * qd[i-7]
        Xup[i] = XJ @ model.Xtree[i]
        v[i] = Xup[i] @ v[model.parent[i]] + vJ
        a[i] = Xup[i] @ a[model.parent[i]] + S[i] * qdd[i-7] + crm(v[i]) @ vJ
        IC[i] = model.I[i]
        pC[i] = IC[i] @ a[i] + crf(v[i]) @ IC[i] @ v[i]

    if f_ext is not None and len(f_ext) > 0:
        prnt = [x - 5 for x in model.parent[6:]]
        pC_list = list(pC.values())[1:]  # Skip first 5
        Xup_list = list(Xup.values())
        f_ext_list = f_ext[6:]
        pC_new = apply_external_forces(prnt, Xup_list, pC_list, f_ext_list)
        for i, p in enumerate(pC_new, 6):
            pC[i] = p

    for i in range(model.NB, 6, -1):
        parent = model.parent[i]
        IC[parent] = IC[parent] + Xup[i].T @ IC[i] @ Xup[i]
        pC[parent] = pC[parent] + Xup[i].T @ pC[i]

    # Floating-base acceleration without gravity
    a[6] = -np.linalg.solve(IC[6], pC[6])

    tau = np.zeros(model.NB - 6)
    for i in range(7, model.NB + 1):
        a[i] = Xup[i] @ a[model.parent[i]]
        tau[i-7] = S[i].T @ (IC[i] @ a[i] + pC[i])

    qnd = rqd(vfb[0:3], qn)        # derivative of qn
    rd = Vpt(vfb, r)               # lin vel of flt base origin
    # f.b. accn in fixed-base coords
    afb = np.linalg.solve(Xup[6], a[6]) + a_grav

    xdfb = np.concatenate([qnd, rd, afb])

    return xdfb, tau
