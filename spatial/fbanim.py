import numpy as np
from spatial.fbkin import fbkin


def fbanim(X, Qr=None):
    """
    Floating Base Inverse Kinematics for Animation.

    Calculates joint position data for smooth animations by removing discontinuities
    from joint angles when they wrap around or pass through kinematic singularities.

    Parameters:
        X: State vector array (13×N, 13×1×N, 7×N, or 7×1×N) where each column contains
           at least the first 7 elements of a 13-element singularity-free state vector
        Qr: Optional joint position data (M×N or M×1×N) for real joints in mechanism

    Returns:
        Q: Joint position matrix (6×N or (6+M)×N) containing floating base joint data
           and optionally the real joint data

    Note: Algorithm assumes less than π/2 changes between consecutive columns
          (except at singularities). Visible glitches may still occur if violated.
    """

    # Collapse 3D -> 2D array if needed
    if len(X.shape) == 3:
        X = X[:, 0, :]

    # Apply kinematic transform using fbkin
    N = X.shape[1]
    Q = np.zeros((6, N))
    for i in range(N):
        Q[:, i] = fbkin(X[0:7, i])

    # Remove wrap-arounds and step-changes on passing through singularity
    for i in range(1, N):
        # Handle q6 wrapping
        n = round((Q[5, i-1] - Q[5, i]) / np.pi)
        q6 = Q[5, i] + n * np.pi

        # Handle q4 and q6 interaction at singularities
        if Q[4, i] >= 0:
            q46 = Q[3, i] + Q[5, i]
            q46 = q46 + 2*np.pi * \
                round((Q[3, i-1] + Q[5, i-1] - q46) / (2*np.pi))
            Q[3, i] = q46 - q6
        else:
            q46 = Q[3, i] - Q[5, i]
            q46 = q46 + 2*np.pi * \
                round((Q[3, i-1] - Q[5, i-1] - q46) / (2*np.pi))
            Q[3, i] = q46 + q6

        Q[5, i] = q6

        # Handle q5 at singularities
        if n % 2 == 0:
            q5 = Q[4, i]
        else:
            q5 = np.pi - Q[4, i]

        Q[4, i] = q5 + 2*np.pi * round((Q[4, i-1] - q5) / (2*np.pi))

    # Add real joint data if provided
    if Qr is not None:
        if len(Qr.shape) == 3:
            Qr = Qr[:, 0, :]
        Q = np.vstack([Q, Qr])

    return Q
