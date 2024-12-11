import numpy as np
from spatial.rotx import rotx
from spatial.roty import roty
from spatial.rotz import rotz
from spatial.xlt import xlt
from spatial.plnr import plnr


def jcalc(jtyp, q):
    """
    Calculate joint transform and motion subspace matrices.

    Parameters:
        jtyp: Either a string or a dict containing joint type code.
              For parameterized joints (e.g. helical), jtyp must be a dict
              with 'code' and 'pars' fields.
        q: Joint position variable

    Returns:
        Xj: Joint transform matrix
        S: Motion subspace matrix

    Joint types:
        'Rx': Revolute X axis
        'Ry': Revolute Y axis
        'R','Rz': Revolute Z axis
        'Px': Prismatic X axis
        'Py': Prismatic Y axis
        'P','Pz': Prismatic Z axis
        'H': Helical (Z axis)
        'r': Planar revolute
        'px': Planar prismatic X axis
        'py': Planar prismatic Y axis
    """

    # Get joint type code
    if isinstance(jtyp, str):
        code = jtyp
    else:
        code = jtyp['code']

    # Calculate joint transform and motion subspace based on joint type
    if code == 'Rx':  # revolute X axis
        Xj = rotx(q)
        S = np.array([[1], [0], [0], [0], [0], [0]])

    elif code == 'Ry':  # revolute Y axis
        Xj = roty(q)
        S = np.array([[0], [1], [0], [0], [0], [0]])

    elif code in ['R', 'Rz']:  # revolute Z axis
        Xj = rotz(q)
        S = np.array([[0], [0], [1], [0], [0], [0]])

    elif code == 'Px':  # prismatic X axis
        Xj = xlt([q, 0, 0])
        S = np.array([[0], [0], [0], [1], [0], [0]])

    elif code == 'Py':  # prismatic Y axis
        Xj = xlt([0, q, 0])
        S = np.array([[0], [0], [0], [0], [1], [0]])

    elif code in ['P', 'Pz']:  # prismatic Z axis
        Xj = xlt([0, 0, q])
        S = np.array([[0], [0], [0], [0], [0], [1]])

    elif code == 'H':  # helical (Z axis)
        pitch = jtyp['pars']['pitch']
        Xj = rotz(q) @ xlt([0, 0, q * pitch])
        S = np.array([[0], [0], [1], [0], [0], [pitch]])

    elif code == 'r':  # planar revolute
        Xj = plnr(q, [0, 0])
        S = np.array([[1], [0], [0]])

    elif code == 'px':  # planar prismatic X axis
        Xj = plnr(0, [q, 0])
        S = np.array([[0], [1], [0]])

    elif code == 'py':  # planar prismatic Y axis
        Xj = plnr(0, [0, q])
        S = np.array([[0], [0], [1]])

    else:
        raise ValueError(f"Unrecognized joint code '{code}'")

    return Xj, S
