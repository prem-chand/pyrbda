import numpy as np
from spatial.jcalc import jcalc
from spatial.crm import crm
from spatial.crf import crf
from spatial.get_gravity import get_gravity
from dynamics.continuous_dynamics.CoriolisAndGravityTerms import CoriolisAndGravityTerms
from spatial.InertiaMatrix import InertiaMatrix


def HandC(sys):
    """
    Calculate coefficients of equation of motion.

    Calculates the coefficients of the joint-space equation of motion:
    tau = H(q)qdd + C(q, qd, f_ext), where q, qd, and qdd are the joint
    position, velocity, and acceleration vectors, H is the joint-space
    inertia matrix, C is the vector of gravity, external-force, and
    velocity-product terms, and tau is the joint force vector.

    Parameters:
        sys: System containing model and state information

    Returns:
        H: Joint-space inertia matrix
        C: Vector of gravity, external-force, and velocity-product terms
    """

    model = sys.Model
    Xtree = sys.Model.Xtree

    q = sys.qpos0
    qd = sys.qvel0

    nd = model.params.nb
    body_id = model.params.body_id

    a_grav = get_gravity(sys.Model)

    # Initialize dictionaries
    Xup = {}
    v = {}
    avp = {}
    fvp = {}
    S = {}

    idx1 = slice(0, 6)  # corresponds to free joint
    idx2 = [i+6 for i in range(12)]  # corresponds to revolute joints
    v_idx = [idx1, *idx2]

    idx3 = [i+7 for i in range(13)]  # corresponds to all joints
    q_idx = [slice(0, 7), *idx3]

    parent_dict = model.parent_id
    parent_dict.pop(0)

    # Forward pass for velocities and accelerations
    for i in range(nd-1):
        XJ, S[i] = jcalc(model.jtype[i], q[q_idx[i]])
        vJ = (S[i] @ qd[v_idx[i]]).reshape((-1, 1))
        
        Xup[i] = XJ @ Xtree[i]

        if parent_dict[i+1] == 0:
            v[i] = vJ
            avp[i] = Xup[i] @ (-a_grav)
        else:
            v[i] = Xup[i] @ v[parent_dict[i]] + vJ
            avp[i] = Xup[i] @ avp[parent_dict[i]] + crm(v[i]) @ vJ

        fvp[i] = model.fullinertia[i+1] @ avp[i] + \
            crf(v[i]) @ model.fullinertia[i+1] @ v[i]

    # Calculate C using Coriolis and gravity terms
    C = CoriolisAndGravityTerms(sys, q, S, Xup, fvp)

    # Calculate H using inertia matrix
    H = InertiaMatrix(sys, q, S, Xup)

    return H, C
