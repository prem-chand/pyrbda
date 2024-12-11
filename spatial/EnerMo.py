import numpy as np
from spatial.jcalc import jcalc
from spatial.plnr import plnr
from spatial.rotz import rotz
from spatial.xlt import xlt
from spatial.pluho import pluho
from spatial.mcI import mcI
from spatial.get_gravity import get_gravity


def EnerMo(model, q, qd):
    """
    Calculate energy, momentum and related quantities.

    Parameters:
        model: Robot model structure
        q: Joint position vector
        qd: Joint velocity vector

    Returns:
        ret: Structure containing:
            - KE: Kinetic energy of the system
            - PE: Potential energy of the system
            - htot: Total spatial momentum
            - Itot: Total spatial inertia
            - cm: Position of center of mass
            - vcm: Linear velocity of center of mass
            - Pjt: Joint positions in base coordinates
            - Tr: Transformation matrices
    """

    # Initialize variables
    KE = np.zeros(model.NB)
    Xup = {}
    v = {}
    Ic = {}
    hc = {}
    Xa = {}
    Tr = {}
    Pjt = {}

    # First forward pass for velocities and kinetic energy
    for i in range(1, model.NB + 1):
        XJ, S = jcalc(model.jtype[i], q[i-1])
        vJ = S * qd[i-1]
        Xup[i] = XJ @ model.Xtree[i]

        if model.parent[i] == 0:
            v[i] = vJ
        else:
            v[i] = Xup[i] @ v[model.parent[i]] + vJ

        Ic[i] = model.I[i]
        hc[i] = Ic[i] @ v[i]
        KE[i-1] = 0.5 * v[i].T @ hc[i]

    # Initialize total inertia and momentum
    ret = {}
    ret['Itot'] = np.zeros_like(Ic[1])
    ret['htot'] = np.zeros_like(hc[1])

    # Calculate transformations and joint positions
    for i in range(1, model.NB + 1):
        XJ = jcalc(model.jtype[i], q[i-1])[0]
        Xa[i] = XJ @ model.Xtree[i]

        if model.parent[i] != 0:
            Xa[i] = Xa[i] @ Xa[model.parent[i]]

        if Xa[i].shape[0] == 3:  # planar coordinate transform
            theta, r = plnr(Xa[i])
            X = rotz(theta) @ xlt(np.append(r, 0))
            T = pluho(X)
        else:
            T = pluho(Xa[i])

        # Efficient inverse calculation
        R = T[0:3, 0:3]
        p = T[0:3, 3]
        R_T = R.T
        Tdisp = np.block([
            [R_T, -R_T @ p[:, np.newaxis]],
            [np.zeros((1, 3)), np.ones((1, 1))]
        ])

        Tr[i] = T
        Pjt[i] = Tdisp[0:3, 3]

        # Handle swing foot position
        if i == model.NB:
            Psw_f = np.array([model.l[i], 0, 0, 1])
            Pjt_temp = Tdisp @ Psw_f
            Pjt[i+1] = Pjt_temp[0:3]

    ret['Pjt'] = Pjt
    ret['Tr'] = Tr

    # Backward pass for total inertia and momentum
    for i in range(model.NB, 0, -1):
        if model.parent[i] != 0:
            Ic[model.parent[i]] = Ic[model.parent[i]] + \
                Xup[i].T @ Ic[i] @ Xup[i]
            hc[model.parent[i]] = hc[model.parent[i]] + Xup[i].T @ hc[i]
        else:
            ret['Itot'] = ret['Itot'] + Xup[i].T @ Ic[i] @ Xup[i]
            ret['htot'] = ret['htot'] + Xup[i].T @ hc[i]

    # Calculate gravitational terms
    a_grav = get_gravity(model)

    if len(a_grav) == 6:  # 3D case
        g = a_grav[3:6]  # 3D linear gravitational acceleration
        h = ret['htot'][3:6]  # 3D linear momentum
    else:  # 2D case
        g = a_grav[1:3]  # 2D gravity
        h = ret['htot'][1:3]  # 2D linear momentum

    # Calculate mass properties
    mass, cm = mcI(ret['Itot'])

    # Store final results
    ret['KE'] = np.sum(KE)
    ret['PE'] = -mass * np.dot(cm, g)
    ret['cm'] = cm
    ret['vcm'] = h / mass

    return ret
