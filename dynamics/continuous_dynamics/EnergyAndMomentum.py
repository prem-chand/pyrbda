import numpy as np
from spatial.jcalc import jcalc
from spatial.mcI import mcI
from spatial.get_gravity import get_gravity
from spatial.skew import skew


def EnergyAndMomentum(obj, sys):
    """
    Calculate energy, momentum and related quantities.

    Parameters:
        obj: Object containing necessary methods
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

    q = sys.States.q.sym
    qd = sys.States.dq.sym

    nd = model.nd
    jtype = model.jtype
    parent = model.parent

    # Initialize kinetic energy array
    KE = np.zeros(nd)

    # Initialize dictionaries
    Xup = {}
    v = {}
    Ic = {}
    hc = {}

    # Forward pass for velocities and kinetic energies
    for i in range(1, nd + 1):
        XJ, S = jcalc(jtype[i], q[i-1])
        vJ = S * qd[i-1]
        Xup[i] = XJ @ Xtree[i]

        if parent[i] == 0:
            v[i] = vJ
        else:
            v[i] = Xup[i] @ v[parent[i]] + vJ

        Ic[i] = model.I[i]
        hc[i] = Ic[i] @ v[i]
        KE[i-1] = 0.5 * v[i].T @ hc[i]

    # Initialize total inertia and momentum
    ret = {}
    ret['Itot'] = np.zeros_like(Ic[1])
    ret['htot'] = np.zeros_like(hc[1])

    # Backward pass for total inertia and momentum
    for i in range(nd, 0, -1):
        if parent[i] != 0:
            Ic[parent[i]] = Ic[parent[i]] + Xup[i].T @ Ic[i] @ Xup[i]
            hc[parent[i]] = hc[parent[i]] + Xup[i].T @ hc[i]
        else:
            ret['Itot'] = ret['Itot'] + Xup[i].T @ Ic[i] @ Xup[i]
            ret['htot'] = ret['htot'] + Xup[i].T @ hc[i]

    # Get gravity vector
    a_grav = get_gravity(model)

    if len(a_grav) == 6:  # 3D case
        g = a_grav[3:6]  # 3D linear gravitational acceleration
        h = ret['htot'][3:6]  # 3D linear momentum
    else:  # 2D case
        g = a_grav[1:3]  # 2D gravity
        h = ret['htot'][1:3]  # 2D linear momentum

    # Calculate mass properties
    mass, cm = mcI(ret['Itot'])

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
