import numpy as np
from spatial.jcalc import jcalc
from spatial.crm import crm
from spatial.crf import crf
from spatial.get_gravity import get_gravity
from dynamics.continuous_dynamics.CoriolisAndGravityTerms import CoriolisAndGravityTerms
from spatial.InertiaMatrix import InertiaMatrix


def HandC(obj, sys):
    """
    Calculate coefficients of equation of motion.

    Calculates the coefficients of the joint-space equation of motion:
    tau = H(q)qdd + C(q, qd, f_ext), where q, qd, and qdd are the joint
    position, velocity, and acceleration vectors, H is the joint-space
    inertia matrix, C is the vector of gravity, external-force, and
    velocity-product terms, and tau is the joint force vector.

    Parameters:
        obj: Object containing necessary methods
        sys: System containing model and state information

    Returns:
        H: Joint-space inertia matrix
        C: Vector of gravity, external-force, and velocity-product terms
    """

    model = sys.Model
    Xtree = sys.Model.Xtree

    q = sys.States.q.sym
    qd = sys.States.dq.sym

    nd = model.nd

    a_grav = get_gravity(sys.Model)

    # Initialize dictionaries
    Xup = {}
    v = {}
    avp = {}
    fvp = {}
    S = {}

    # Forward pass for velocities and accelerations
    for i in range(1, nd + 1):
        XJ, S[i] = jcalc(model.jtype[i], q[i-1])
        vJ = S[i] * qd[i-1]
        Xup[i] = XJ @ Xtree[i]

        if model.parent[i] == 0:
            v[i] = vJ
            avp[i] = Xup[i] @ (-a_grav)
        else:
            v[i] = Xup[i] @ v[model.parent[i]] + vJ
            avp[i] = Xup[i] @ avp[model.parent[i]] + crm(v[i]) @ vJ

        fvp[i] = model.I[i] @ avp[i] + crf(v[i]) @ model.I[i] @ v[i]

    # Calculate C using Coriolis and gravity terms
    C = CoriolisAndGravityTerms(obj, sys, q, S, Xup, fvp)

    # Calculate H using inertia matrix
    H = InertiaMatrix(obj, sys, q, S, Xup)

    return H, C
