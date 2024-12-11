import numpy as np
from numpy.linalg import solve


def apply_external_forces(parent, Xup, f_in, f_ext):
    """
    Subtract external forces from a given array of forces.

    Incorporates external forces into dynamics algorithm calculations by
    subtracting transformed external forces from input forces.

    Parameters:
        parent: Parent array of the model
        Xup: Link-to-link coordinate transforms
        f_in: Input forces in link coordinates
        f_ext: External forces in absolute coordinates
            Must be None, empty list, or list of length NB where:
            - f_ext[i] is None/empty for no force on body i
            - f_ext[i] is spatial/planar vector for force on body i

    Returns:
        f_out: Modified forces in link coordinates (f_in - transformed f_ext)
    """

    # Start with copy of input forces
    f_out = f_in.copy() if isinstance(f_in, list) else f_in

    # If no external forces, return original forces
    if f_ext is None or len(f_ext) == 0:
        return f_out

    # Calculate absolute transforms and apply forces
    Xa = {}

    for i in range(1, len(parent) + 1):
        if parent[i] == 0:
            Xa[i] = Xup[i]
        else:
            Xa[i] = Xup[i] @ Xa[parent[i]]

        # Apply external force if it exists for this body
        if f_ext[i-1] is not None and len(f_ext[i-1]) > 0:
            # Transform external force to link coordinates
            # f_out[i] = f_out[i] - Xa[i].T \ f_ext[i-1]
            f_out[i] = f_out[i] - solve(Xa[i].T, f_ext[i-1])

    return f_out
