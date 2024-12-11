import numpy as np


def floatbase(model):
    """
    Construct the floating-base equivalent of a fixed-base model.

    Converts a fixed-base spatial kinematic tree to a floating-base kinematic tree:
    - Old body 1 becomes new body 6 (floating base)
    - Old joint 1 is discarded
    - Six new joints added (3 prismatic, 3 revolute; x,y,z order)
    - Five new zero-mass bodies added (1-5) to connect new joints
    - All other bodies/joints preserved but numbers incremented by 5

    Parameters:
        model: Fixed-base model structure containing at least:
               - NB: Number of bodies
               - jtype: List of joint types
               - parent: List of parent body indices
               - Xtree: List of coordinate transforms
               - I: List of spatial inertias

    Returns:
        fbmodel: Floating-base model with same structure as input

    Caution: Singularity occurs when q[4] = ±π/2
    """

    # Check if model is spatial (not planar)
    if model.Xtree[0].shape[0] == 3:
        raise ValueError('floatbase applies to spatial models only')

    # Check for single fixed base connection
    if any(np.array(model.parent[1:]) == 0):
        raise ValueError('only one connection to a fixed base allowed')

    # Warning for gamma_q field
    if hasattr(model, 'gamma_q'):
        print('Warning: floating a model with gamma_q (joint numbers will change)')

    # Warning for non-identity Xtree[0]
    if not np.array_equal(model.Xtree[0], np.eye(6)):
        print('Warning: Xtree[0] not identity')

    # Create new model (shallow copy)
    fbmodel = type('Model', (), dict(model.__dict__))

    # Update number of bodies
    fbmodel.NB = model.NB + 5

    # Update joint types
    fbmodel.jtype = ['Px', 'Py', 'Pz', 'Rx', 'Ry', 'Rz'] + model.jtype[1:]

    # Update parent array
    fbmodel.parent = [0] + list(range(1, 5)) + [i + 5 for i in model.parent]

    # Update coordinate transforms
    eye6 = np.eye(6)
    fbmodel.Xtree = [eye6] * 5 + model.Xtree

    # Update spatial inertias
    zeros6 = np.zeros((6, 6))
    fbmodel.I = [zeros6] * 5 + model.I

    # Update appearance if present
    if hasattr(model, 'appearance'):
        fbmodel.appearance = type('Appearance', (), {})
        fbmodel.appearance.body = [{}] * 5 + model.appearance.body

    # Update camera if present
    if hasattr(model, 'camera') and hasattr(model.camera, 'body'):
        fbmodel.camera = type('Camera', (), {})
        if model.camera.body > 0:
            fbmodel.camera.body = model.camera.body + 5

    # Update ground contact if present
    if hasattr(model, 'gc'):
        fbmodel.gc = type('GC', (), {})
        fbmodel.gc.body = [b + 5 for b in model.gc.body]

    return fbmodel
