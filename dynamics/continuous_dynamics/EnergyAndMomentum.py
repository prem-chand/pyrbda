import numpy as np
from spatial.jcalc import jcalc
from spatial.mcI import mcI
from spatial.get_gravity import get_gravity
from spatial.skew import skew


def EnergyAndMomentum(sys):
    """
    Calculate energy, momentum and related quantities.

    Parameters:
        sys: System containing model and state information

    Returns:
        KE: Kinetic energy of the system
        PE: Potential energy of the system
        cm: Position of center of mass
        vcm: Linear velocity of center of mass
        cam: Centroidal angular momentum
    """

    model = sys.Model
    Xtree = model.Xtree

    q = sys.qpos0
    qd = sys.qvel0

    nd = model.params.nb
    jtype = model.jtype
    parent = model.parent
    parent_id = model.parent_id

    idx1 = slice(0, 6)  # corresponds to free joint
    idx2 = [i + 6 for i in range(12)]  # corresponds to revolute joints
    v_idx = [idx1, *idx2]

    idx3 = [i + 7 for i in range(13)]  # corresponds to all joints
    q_idx = [slice(0, 7), *idx3]

    # Initialize kinetic energy array
    KE = np.zeros(nd)

    # Initialize dictionaries
    Xup = {}
    v = {}
    Ic = {}
    hc = {}
    S = {}

    # Forward pass for velocities and kinetic energies
    for i in range(nd-1):
        XJ, S[i] = jcalc(model.jtype[i], q[q_idx[i]])
        vJ = (S[i] @ qd[v_idx[i]]).reshape((-1, 1))
        Xup[i] = XJ @ Xtree[i]

        if parent_id[i+1] == 0:
            v[i] = vJ
        else:
            v[i] = Xup[i] @ v[parent_id[i]] + vJ

        Ic[i] = model.fullinertia[i+1]
        hc[i] = Ic[i] @ v[i]
        KE[i-1] = 0.5 * v[i].T @ hc[i]

    # Initialize total inertia and momentum
    ret = {}
    ret['Itot'] = np.zeros_like(Ic[1])
    ret['htot'] = np.zeros_like(hc[1])

    # Backward pass for total inertia and momentum
    for i in range(nd-1, 0, -1):
        if parent_id[i] != 0:
            Ic[parent_id[i]-1] = Ic[parent_id[i]-1] + \
                Xup[i-1].T @ Ic[i-1] @ Xup[i-1]
            hc[parent_id[i]-1] = hc[parent_id[i]-1] + Xup[i-1].T @ hc[i-1]
        else:
            ret['Itot'] = ret['Itot'] + Xup[i-1].T @ Ic[i-1] @ Xup[i-1]
            ret['htot'] = ret['htot'] + Xup[i-1].T @ hc[i-1]

    # Get gravity vector
    a_grav = get_gravity(model)

    if len(a_grav) == 6:  # 3D case
        g = a_grav[3:6]  # 3D linear gravitational acceleration
        h = ret['htot'][3:6]  # 3D linear momentum
    else:  # 2D case
        g = a_grav[1:3]  # 2D gravity
        h = ret['htot'][1:3]  # 2D linear momentum

    # Calculate mass properties
    mass, cm, II = mcI(ret['Itot'])

    # Calculate energies and velocities
    KE = np.sum(KE)
    PE = -mass * np.dot(cm, g)
    vcm = h / mass

    # Calculate centroidal momentum
    p0G = cm
    X0G = np.block([
        [np.eye(3), np.zeros((3, 3))],
        [skew(p0G), np.eye(3)]
    ])
    hG = X0G.T @ ret['htot']

    cam = hG

    return KE, PE, cm, vcm, cam
