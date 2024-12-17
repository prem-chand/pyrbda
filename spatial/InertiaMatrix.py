import numpy as np


def InertiaMatrix(sys, q, S, Xup):
    """
    Calculate the joint-space inertia matrix.

    Parameters:
        obj: Object containing necessary methods
        sys: System containing model information
        q: Joint position vector
        S: Dictionary of motion subspaces
        Xup: Dictionary of coordinate transforms

    Returns:
        H: Joint-space inertia matrix
    """

    model = sys.Model

    # Initialize composite inertia calculation
    IC = model.fullinertia.copy()  # Make a copy to avoid modifying original

    nd = model.params.nj  # number of joints
    nv = model.params.nv
    parent = model.parent

    # IC = {}

    idx1 = slice(0, 6)  # corresponds to free joint
    idx2 = [i+6 for i in range(12)]  # corresponds to revolute joints
    v_idx = [idx1, *idx2]

    # Backward pass to compute composite inertias
    for i in range(nd, 0, -1):
        if model.parent_id[i] != 0:
            IC[model.parent_id[i]] = IC[model.parent_id[i]] + \
                Xup[i].T @ IC[i] @ Xup[i]

    # # Backward pass to compute composite inertias
    # for i in range(nd, 0, -1):
    #     body_name = model.body_names[i]
    #     parent_id = model.params.body_id[model.parent[body_name]]
    #     if model.parent[body_name] != 'world':
    #         IC[parent_id] = IC[parent_id] + Xup[i].T @ IC[i] @ Xup[i]

    # Initialize inertia matrix
    if isinstance(q, np.ndarray):
        H = np.zeros((nv, nv))
    else:
        H = 0*sys.States['dq']@sys.States['dq'].T

    # Forward pass to compute inertia matrix
    # fh1 = IC[1] @ S[1]
    # H[:6, :6] = S[1].T @ fh1
    for i in range(1, nd + 1):
        body_i = model.body_names[i]
        fh = IC[i] @ S[i]
        H[v_idx[i-1], v_idx[i-1]] = S[i].T @ fh

        j = i
        body_j = model.body_names[j]

        while model.parent_id[j] > 0:
            fh = Xup[j].T @ fh
            j = model.parent_id[j]
            H[v_idx[i-1], v_idx[j-1]] = (S[j].T @ fh).reshape(-1)
            H[v_idx[j-1], v_idx[i-1]] = H[v_idx[i-1],
                                          v_idx[j-1]]  # Symmetric matrix

    return H


# import numpy as np


# def InertiaMatrix(sys, q, S, Xup):
#     """
#     Calculate the joint-space inertia matrix.

#     Parameters:
#         obj: Object containing necessary methods
#         sys: System containing model information
#         q: Joint position vector
#         S: Dictionary of motion subspaces
#         Xup: Dictionary of coordinate transforms

#     Returns:
#         H: Joint-space inertia matrix
#     """

#     model = sys.Model

#     # Initialize composite inertia calculation
#     IC = model.I.copy()  # Make a copy to avoid modifying original

#     nd = model.nd
#     parent = model.parent

#     # Backward pass to compute composite inertias
#     for i in range(nd, 0, -1):
#         if parent[i] != 0:
#             IC[parent[i]] = IC[parent[i]] + Xup[i].T @ IC[i] @ Xup[i]

#     # Initialize inertia matrix
#     H = np.zeros((nd, nd))

#     # Forward pass to compute inertia matrix
#     for i in range(1, nd + 1):
#         fh = IC[i] @ S[i]
#         H[i-1, i-1] = S[i].T @ fh

#         j = i
#         while parent[j] > 0:
#             fh = Xup[j].T @ fh
#             j = parent[j]
#             H[i-1, j-1] = S[j].T @ fh
#             H[j-1, i-1] = H[i-1, j-1]  # Symmetric matrix

#     return H


def InertiaMatrix(sys, q, S, Xup):
    """
    Calculate the joint-space inertia matrix.

    Parameters:
        obj: Object containing necessary methods
        sys: System containing model information
        q: Joint position vector
        S: Dictionary of motion subspaces
        Xup: Dictionary of coordinate transforms

    Returns:
        H: Joint-space inertia matrix
    """

    model = sys.Model

    # Initialize composite inertia calculation
    IC = model.fullinertia.copy()  # Make a copy to avoid modifying original

    nd = model.params.nj  # number of joints
    nv = model.params.nv
    parent = model.parent

    # IC = {}

    idx1 = slice(0, 6)  # corresponds to free joint
    idx2 = [i+6 for i in range(12)]  # corresponds to revolute joints
    v_idx = [idx1, *idx2]

    # Backward pass to compute composite inertias
    for i in range(nd, 0, -1):
        body_name = model.body_names[i]
        parent_id = model.params.body_id[model.parent[body_name]]
        if model.parent[body_name] != 'world':
            IC[parent_id] = IC[parent_id] + Xup[i-1].T @ IC[i] @ Xup[i-1]

    # Initialize inertia matrix
    if isinstance(q, np.ndarray):
        H = np.zeros((nv, nv))
    else:
        H = 0*sys.States['dq']@sys.States['dq'].T

    # Forward pass to compute inertia matrix
    # fh1 = IC[1] @ S[1]
    # H[:6, :6] = S[1].T @ fh1
    for i in range(1, nd + 1):
        body_i = model.body_names[i]
        fh = IC[i] @ S[i-1]
        H[v_idx[i-1], v_idx[i-1]] = S[i-1].T @ fh

        j = i
        body_j = model.body_names[j]

        while model.parent_id[j] > 1:
            fh = Xup[j-1].T @ fh
            j = model.parent_id[j]
            H[v_idx[i-1], v_idx[j-1]] = (S[j-1].T @ fh).reshape(-1)
            H[v_idx[j-1], v_idx[i-1]] = H[v_idx[i-1],
                                          v_idx[j-1]]  # Symmetric matrix

    return H


# import numpy as np


# def InertiaMatrix(sys, q, S, Xup):
#     """
#     Calculate the joint-space inertia matrix.

#     Parameters:
#         obj: Object containing necessary methods
#         sys: System containing model information
#         q: Joint position vector
#         S: Dictionary of motion subspaces
#         Xup: Dictionary of coordinate transforms

#     Returns:
#         H: Joint-space inertia matrix
#     """

#     model = sys.Model

#     # Initialize composite inertia calculation
#     IC = model.I.copy()  # Make a copy to avoid modifying original

#     nd = model.nd
#     parent = model.parent

#     # Backward pass to compute composite inertias
#     for i in range(nd, 0, -1):
#         if parent[i] != 0:
#             IC[parent[i]] = IC[parent[i]] + Xup[i].T @ IC[i] @ Xup[i]

#     # Initialize inertia matrix
#     H = np.zeros((nd, nd))

#     # Forward pass to compute inertia matrix
#     for i in range(1, nd + 1):
#         fh = IC[i] @ S[i]
#         H[i-1, i-1] = S[i].T @ fh

#         j = i
#         while parent[j] > 0:
#             fh = Xup[j].T @ fh
#             j = parent[j]
#             H[i-1, j-1] = S[j].T @ fh
#             H[j-1, i-1] = H[i-1, j-1]  # Symmetric matrix

#     return H
