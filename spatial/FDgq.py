import numpy as np
from dynamics.continuous_dynamics.HandC import HandC


def FDgq(model, q, qd, tau, f_ext=None):
    """
    Forward Dynamics via CRBA + constraint function gamma_q.

    Calculates forward dynamics of a kinematic tree subject to kinematic
    constraints defined in model.gamma_q.

    Parameters:
        model: Robot model structure with gamma_q method
        q: Joint position vector
        qd: Joint velocity vector
        tau: Joint force vector
        f_ext: Optional external forces

    Returns:
        qdd: Joint acceleration vector with constraint stabilization

    Note: q and qd don't need to satisfy constraints exactly but should be close.
          qdd typically includes constraint-stabilization component.
    """

    # Apply kinematic constraints
    q, qd, G, g = model.gamma_q(model, q, qd)

    # Calculate mass matrix and bias forces
    if f_ext is None:
        H, C = HandC(model, q, qd)
    else:
        H, C = HandC(model, q, qd, f_ext)

    # Calculate constrained acceleration (eq 3.20 in RBDA)
    # qdd = G * ((G'*H*G) \ (G'*(tau-C-H*g))) + g
    temp = G.T @ H @ G
    temp2 = G.T @ (tau - C - H @ g)
    qdd = G @ np.linalg.solve(temp, temp2) + g

    return qdd
