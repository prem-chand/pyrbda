import numpy as np


def skew(v):
    """
    Convert a 3D vector to a skew-symmetric matrix.
    """
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])


def pluho(matrix_in):
    """
    Convert between Plucker and 4x4 homogeneous coordinate transform.

    Parameters:
        matrix_in: Either a 6x6 Plucker matrix (X) or a 4x4 homogeneous transform matrix (T)

    Returns:
        If input is 6x6: returns 4x4 homogeneous transform matrix
        If input is 4x4: returns 6x6 Plucker matrix
    """
    matrix_in = np.asarray(matrix_in)

    if matrix_in.shape == (6, 6):  # Plucker -> 4x4 homogeneous
        E = matrix_in[0:3, 0:3]
        mErx = matrix_in[3:6, 0:3]  # - E r cross
        in2skew = mErx @ E.T

        if in2skew.shape == (3, 3):
            out2skew = 0.5 * np.array([
                in2skew[2, 1] - in2skew[1, 2],
                in2skew[0, 2] - in2skew[2, 0],
                in2skew[1, 0] - in2skew[0, 1]
            ])

        out = np.vstack([
            np.hstack([E, out2skew.reshape(3, 1)]),
            np.array([[0, 0, 0, 1]])
        ])

    else:  # 4x4 homogeneous -> Plucker
        E = matrix_in[0:3, 0:3]
        mEr = matrix_in[0:3, 3]  # - E r
        out = np.vstack([
            np.hstack([E, np.zeros((3, 3))]),
            np.hstack([skew(mEr) @ E, E])
        ])

    return out
