import numpy as np


def mpyH(H, lambda_arr, x):
    """
    Calculate H*x exploiting branch-induced sparsity in H.

    Computes H*x where x is a vector and H is a symmetric, positive-definite matrix.
    The nonzero elements on row i below the main diagonal appear only in columns 
    lambda(i), lambda(lambda(i)), etc. This is the pattern of branch-induced sparsity.
    H and lambda can be regarded as the joint-space inertia matrix and parent array 
    of a kinematic tree.

    Parameters:
        H: Symmetric positive-definite matrix with branch-induced sparsity
        lambda_arr: Parent array (must satisfy 0 <= lambda[i] < i for all i)
        x: Input vector to multiply

    Returns:
        y: Result of H*x multiplication
    """
    n = H.shape[0]
    y = x.copy()  # Create copy to preserve input vector

    # Diagonal terms
    for i in range(n):
        y[i] = H[i, i] * x[i]

    # Off-diagonal terms
    for i in range(n-1, -1, -1):  # n-1 down to 0
        j = lambda_arr[i]
        while j != 0:
            y[i] = y[i] + H[i, j] * x[j]
            y[j] = y[j] + H[i, j] * x[i]
            j = lambda_arr[j]

    return y
