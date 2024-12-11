import numpy as np


def rqd(in1, in2):
    """
    rqd  derivative of unit quaternion from angular velocity
    qd=rqd(wA,q) and qd=rqd(q,wB) calculate the derivative of a unit
    quaternion, q, representing the orientation of a coordinate frame B
    relative to frame A, given the angular velocity w of B relative to A. If
    w is expressed in A coordinates then use rqd(wA,q); and if w is expressed
    in B coordinates then use rqd(q,wB). If the length of the first argument
    is 4 then it is assumed to be q, otherwise it is assumed to be wA. The
    return value is a column vector, but the arguments can be row or column
    vectors. It is not necessary for |q| to be exactly 1. If |q|~=1 then qd
    contains a magnitude-stabilizing term that will cause |q| to converge
    towards 1 if q is obtained by numerical integration of qd.
    """
    Kstab = 0.1  # magnitude stabilization constant: value not critical, but K>1 too big

    in1 = np.asarray(in1).flatten()
    in2 = np.asarray(in2).flatten()

    if len(in1) == 4:  # arguments are q and wB
        q = in1
        w = in2
        Q = np.array([[q[0], -q[1], -q[2], -q[3]],
                      [q[1],  q[0], -q[3],  q[2]],
                      [q[2],  q[3],  q[0], -q[1]],
                      [q[3], -q[2],  q[1],  q[0]]])
    else:  # arguments are wA and q
        q = in2
        w = in1
        Q = np.array([[q[0], -q[1], -q[2], -q[3]],
                      [q[1],  q[0],  q[3], -q[2]],
                      [q[2], -q[3],  q[0],  q[1]],
                      [q[3],  q[2], -q[1],  q[0]]])

    w = w.reshape(3, 1)
    qd = 0.5 * Q @ np.vstack((Kstab * np.linalg.norm(w)
                             * (1 - np.linalg.norm(q)), w))
    return qd.flatten()
