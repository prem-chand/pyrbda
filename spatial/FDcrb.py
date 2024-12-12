import numpy as np
from dynamics.continuous_dynamics.HandC import HandC


def FDcrb(model, q, qd, tau, f_ext=None):
    """
    Forward Dynamics via Composite-Rigid-Body Algorithm.

    Calculates forward dynamics of a kinematic tree using the 
    composite-rigid-body algorithm.

    Parameters:
        model: Robot model structure
        q: Joint position vector
        qd: Joint velocity vector
        tau: Joint force vector
        f_ext: Optional external forces

    Returns:
        qdd: Joint acceleration vector
    """

    # Calculate mass matrix and bias forces
    if f_ext is None:
        H, C = HandC(model, q, qd)
    else:
        H, C = HandC(model, q, qd, f_ext)

    # Solve for accelerations: H*qdd = tau - C
    qdd = np.linalg.solve(H, tau - C)

    return qdd
