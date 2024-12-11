import numpy as np


def rv(in_array):
    """
    rv  3D rotation vector <--> 3x3 coordinate rotation matrix
    E=rv(v) and v=rv(E) convert between a rotation vector v, whose magnitude
    and direction describe the angle and axis of rotation of a coordinate
    frame B relative to frame A, and the 3x3 coordinate rotation matrix E
    that transforms from A to B coordinates. For example, if v=[theta;0;0]
    then rv(v) produces the same matrix as rx(theta). If the argument is a
    3x3 matrix then it is assumed to be E, otherwise it is assumed to be v.
    rv(E) expects E to be accurately orthonormal, and returns a column vector
    with a magnitude in the range [0,pi]. If the magnitude is exactly pi
    then the sign of the return value is unpredictable, since pi*u and -pi*u,
    where u is any unit vector, both represent the same rotation. rv(v) will
    accept a row or column vector of any magnitude.
    """
    in_array = np.asarray(in_array)

    if in_array.shape == (3, 3):
        return Etov(in_array)
    else:
        return vtoE(in_array.flatten())


def vtoE(v):
    theta = np.linalg.norm(v)
    if theta == 0:
        return np.eye(3)
    else:
        s = np.sin(theta)
        c = np.cos(theta)
        c1 = 2 * np.sin(theta / 2) ** 2  # 1-cos(h) == 2sin^2(h/2)
        u = v / theta
        return c * np.eye(3) - s * skew(u) + c1 * np.outer(u, u)


def Etov(E):
    w = -skew(E)  # w == s/theta * v
    s = np.linalg.norm(w)
    c = (np.trace(E) - 1) / 2
    theta = np.arctan2(s, c)

    if s == 0:
        return np.array([0, 0, 0])
    elif theta < 0.9 * np.pi:  # a somewhat arbitrary threshold
        return theta / s * w
    else:
        E = E - c * np.eye(3)
        E = E + E.T
        if E[0, 0] >= E[1, 1] and E[0, 0] >= E[2, 2]:
            v = E[:, 0] if w[0] >= 0 else -E[:, 0]
        elif E[1, 1] >= E[2, 2]:
            v = E[:, 1] if w[1] >= 0 else -E[:, 1]
        else:
            v = E[:, 2] if w[2] >= 0 else -E[:, 2]
        return theta / np.linalg.norm(v) * v


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
