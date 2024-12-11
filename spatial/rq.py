import numpy as np


def rq(in_array):
    """
    rq  unit quaternion <--> 3x3 coordinate rotation matrix
    E=rq(q) and q=rq(E) convert between a unit quaternion q, representing
    the orientation of a coordinate frame B relative to frame A, and the 3x3
    coordinate rotation matrix E that transforms from A to B coordinates.
    For example, if B is rotated relative to A about their common X axis by
    an angle h, then q=[cos(h/2);sin(h/2);0;0] and rq(q) produces the same
    matrix as rx(h). If the argument is a 3x3 matrix then it is assumed to
    be E, otherwise it is assumed to be q. rq(E) expects E to be accurately
    orthonormal, and returns a quaternion in a 4x1 matrix; but rq(q) accepts
    any nonzero quaternion, contained in either a row or a column vector, and
    normalizes it before use. As both q and -q represent the same rotation,
    rq(E) returns the value that satisfies q(1)>0. If q(1)==0 then it picks
    the value such that the largest-magnitude element is positive. In the
    event of a tie, the smaller index wins.
    """
    in_array = np.asarray(in_array)

    if in_array.shape == (3, 3):
        return Etoq(in_array)
    else:
        return qtoE(in_array.flatten())


def qtoE(q):
    q = q / np.linalg.norm(q)

    q0s = q[0] * q[0]
    q1s = q[1] * q[1]
    q2s = q[2] * q[2]
    q3s = q[3] * q[3]
    q01 = q[0] * q[1]
    q02 = q[0] * q[2]
    q03 = q[0] * q[3]
    q12 = q[1] * q[2]
    q13 = q[1] * q[3]
    q23 = q[2] * q[3]

    E = 2 * np.array([[q0s + q1s - 0.5, q12 + q03, q13 - q02],
                      [q12 - q03, q0s + q2s - 0.5, q23 + q01],
                      [q13 + q02, q23 - q01, q0s + q3s - 0.5]])
    return E


def Etoq(E):
    tr = np.trace(E)  # trace is 4*q0^2-1
    v = -skew(E)  # v is 2*q0 * [q1;q2;q3]

    if tr > 0:
        q = np.hstack(((tr + 1) / 2, v))
    else:
        E = E - (tr - 1) / 2 * np.eye(3)
        E = E + E.T
        if E[0, 0] >= E[1, 1] and E[0, 0] >= E[2, 2]:
            q = np.hstack((2 * v[0], E[:, 0]))
        elif E[1, 1] >= E[2, 2]:
            q = np.hstack((2 * v[1], E[:, 1]))
        else:
            q = np.hstack((2 * v[2], E[:, 2]))
        if q[0] < 0:
            q = -q

    q = q / np.linalg.norm(q)
    return q


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
