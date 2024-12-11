import numpy as np
from spatial.rq import rq
from spatial.rqd import rqd


def fbkin(i1, i2=None, i3=None):
    """
    Forward and Inverse Kinematics of Floating Base.

    Multiple usage modes:
    1. [x,xd] = fbkin(q,qd,qdd): Forward kinematics
    2. [q,qd,qdd] = fbkin(x,xd): Inverse kinematics
    3. x = fbkin(q,qd) or [q,qd] = fbkin(x): Without acceleration
    4. p = fbkin(q) or q = fbkin(p): Position only

    Parameters:
        i1: First input (q, x, or p)
        i2: Optional second input (qd or xd)
        i3: Optional third input (qdd)

    Returns:
        Various outputs depending on input mode

    Note: Singularities occur when q[4] = ±π/2
    """
    if len(i1) in [13, 7]:
        if i2 is not None and i3 is None:
            return invkin(i1, i2)
        elif i2 is None and i3 is None:
            return invkin(i1)
    else:
        if i2 is not None and i3 is not None:
            return fwdkin(i1, i2, i3)
        elif i2 is not None and i3 is None:
            return fwdkin(i1, i2)
        else:
            return fwdkin(i1)


def fwdkin(q, qd=None, qdd=None):
    """Forward kinematics calculation."""
    c4, s4 = np.cos(q[3]), np.sin(q[3])
    c5, s5 = np.cos(q[4]), np.sin(q[4])
    c6, s6 = np.cos(q[5]), np.sin(q[5])

    E = np.array([
        [c5*c6, c4*s6+s4*s5*c6, s4*s6-c4*s5*c6],
        [-c5*s6, c4*c6-s4*s5*s6, s4*c6+c4*s5*s6],
        [s5, -s4*c5, c4*c5]
    ])

    qn = rq(E)  # unit quaternion fixed-->floating
    r = q[0:3]  # position of floating-base origin

    x = np.zeros(13)
    x[0:4] = qn
    x[4:7] = r

    if qd is not None:
        S = np.array([
            [1, 0, s5],
            [0, c4, -s4*c5],
            [0, s4, c4*c5]
        ])

        omega = S @ qd[3:6]
        rd = qd[0:3]  # lin vel of floating-base origin
        v = np.concatenate([omega, rd + np.cross(r, omega)])  # spatial vel
        x[7:13] = v

        if qdd is not None:
            c4d = -s4*qd[3]
            s4d = c4*qd[3]
            c5d = -s5*qd[4]
            s5d = c5*qd[4]

            Sd = np.array([
                [0, 0, s5d],
                [0, c4d, -s4d*c5-s4*c5d],
                [0, s4d, c4d*c5+c4*c5d]
            ])

            omegad = S @ qdd[3:6] + Sd @ qd[3:6]
            rdd = qdd[0:3]
            a = np.concatenate([
                omegad,
                rdd + np.cross(rd, omega) + np.cross(r, omegad)
            ])

            xd = np.zeros(13)
            xd[0:4] = rqd(omega, rq(E))
            xd[4:7] = rd
            xd[7:13] = a
            return x, xd

    return x


def invkin(x, xd=None):
    """Inverse kinematics calculation."""
    E = rq(x[0:4])  # coord xfm fixed-->floating
    r = x[4:7]  # position of floating-base origin

    q = np.zeros(6)
    q[0:3] = r

    q[4] = np.arctan2(E[2, 0], np.sqrt(E[0, 0]**2 + E[1, 0]**2))
    q[5] = np.arctan2(-E[1, 0], E[0, 0])

    if E[2, 0] > 0:
        q[3] = np.arctan2(E[1, 2]+E[0, 1], E[1, 1]-E[0, 2]) - q[5]
    else:
        q[3] = np.arctan2(E[1, 2]-E[0, 1], E[1, 1]+E[0, 2]) + q[5]

    # Normalize q[3] to [-π, π]
    q[3] = (q[3] + np.pi) % (2*np.pi) - np.pi

    if xd is None:
        return q

    # Calculate velocities
    c4, s4 = np.cos(q[3]), np.sin(q[3])
    c5, s5 = np.cos(q[4]), np.sin(q[4])

    S = np.array([
        [1, 0, s5],
        [0, c4, -s4*c5],
        [0, s4, c4*c5]
    ])

    omega = x[7:10]
    rd = x[10:13] - np.cross(r, omega)

    qd = np.zeros(6)
    qd[0:3] = rd
    qd[3:6] = np.linalg.solve(S, omega)  # will fail at singularity

    if len(xd) == 0:
        return q, qd

    # Calculate accelerations
    c4d = -s4*qd[3]
    s4d = c4*qd[3]
    c5d = -s5*qd[4]
    s5d = c5*qd[4]

    Sd = np.array([
        [0, 0, s5d],
        [0, c4d, -s4d*c5-s4*c5d],
        [0, s4d, c4d*c5+c4*c5d]
    ])

    omegad = xd[7:10]
    rdd = xd[10:13] - np.cross(rd, omega) - np.cross(r, omegad)

    qdd = np.zeros(6)
    qdd[0:3] = rdd
    qdd[3:6] = np.linalg.solve(S, omegad - Sd @ qd[3:6])

    return q, qd, qdd
