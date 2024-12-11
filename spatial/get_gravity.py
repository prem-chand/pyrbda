import numpy as np


def get_gravity(model):
    """
    Get spatial/planar gravitational acceleration vector for given model.

    Computes gravitational acceleration vector for dynamics calculations.
    Returns either spatial or planar vector based on model type.
    Uses model.gravity if present, otherwise defaults to:
    - [0,0,-9.81] for spatial models
    - [0,0] for planar models

    Parameters:
        model: Model structure containing at least:
               - Xtree: List of transforms (used to determine if model is planar)
               - gravity: Optional field specifying gravitational acceleration

    Returns:
        a_grav: Gravitational acceleration vector (spatial or planar)
    """

    # Get gravity vector from model or use default
    if hasattr(model, 'gravity'):
        g = np.asarray(model.gravity).flatten()
    else:
        g = np.array([0, 0, -9.81])

    # Check if model is planar (3x3 transform) or spatial (6x6 transform)
    if model.Xtree[0].shape[0] == 3:  # planar model
        a_grav = np.array([0, g[0], g[1]])
    else:  # spatial model
        a_grav = np.array([0, 0, 0, g[0], g[1], g[2]])

    return a_grav
