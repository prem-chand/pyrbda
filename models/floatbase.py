import numpy as np


def floatbase(model):
    """
    Construct the floating-base equivalent of a fixed-base model.

    Parameters:
        model: A dictionary representing the fixed-base spatial kinematic tree.

    Returns:
        fbmodel: A dictionary representing the floating-base kinematic tree.
    """

    # Check if the model is a spatial model
    if model['Xtree'][0].shape[0] == 3:
        raise ValueError('floatbase applies to spatial models only')

    # Check for multiple connections to a fixed base
    if any(parent == 0 for parent in model['parent'][1:model['NB']]):
        raise ValueError('only one connection to a fixed base allowed')

    # Warning for gamma_q field
    if 'gamma_q' in model:
        print('Warning: floating a model with gamma_q (joint numbers will change)')

    # Check if Xtree{1} is identity
    if not np.array_equal(model['Xtree'][0], np.eye(6)):
        print('Warning: Xtree{1} not identity')

    # Create the floating-base model
    fbmodel = model.copy()
    fbmodel['NB'] = model['NB'] + 5

    # Add new joint types
    fbmodel['jtype'] = ['Px', 'Py', 'Pz',
                        'Rx', 'Ry', 'Rz'] + model['jtype'][1:]

    # Update parent indices
    fbmodel['parent'] = [0, 1, 2, 3, 4] + [p + 5 for p in model['parent']]

    # Initialize Xtree and inertia matrices
    fbmodel['Xtree'] = [np.eye(6)] * 5 + model['Xtree']
    fbmodel['I'] = [np.zeros((6, 6))] * 5 + model['I']

    # Handle appearance field
    if 'appearance' in model:
        fbmodel['appearance'] = {'body': [None]
                                 * 5 + model['appearance']['body']}

    # Handle camera field
    if 'camera' in model and 'body' in model['camera']:
        if model['camera']['body'] > 0:
            fbmodel['camera'] = {'body': model['camera']['body'] + 5}

    # Handle gc field
    if 'gc' in model:
        fbmodel['gc'] = {'body': model['gc']['body'] + 5}

    return fbmodel
