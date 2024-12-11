import numpy as np


def LTL(H, lambda_arr):
    """
    Factorize H -> L'*L exploiting branch-induced sparsity.

    Computes lower-triangular matrix L satisfying L'*L = H, where H is a symmetric,
    positive-definite matrix with branch-induced sparsity pattern determined by lambda.

    Parameters:
        H: Symmetric positive-definite matrix (e.g., joint-space inertia matrix)
        lambda_arr: Parent array satisfying 0 <= lambda[i] < i for all i

    Returns:
        L: Lower-triangular matrix such that L.T @ L = H
    """

    # Work on a copy of H to avoid modifying the input
    H = H.copy()
    n = H.shape[0]

    # Main factorization loop
    for k in range(n-1, -1, -1):
        H[k, k] = np.sqrt(H[k, k])

        # Update columns below diagonal
        i = lambda_arr[k]
        while i != 0:
            H[k, i] = H[k, i] / H[k, k]
            i = lambda_arr[i]

        # Update remaining elements
        i = lambda_arr[k]
        while i != 0:
            j = i
            while j != 0:
                H[i, j] = H[i, j] - H[k, i] * H[k, j]
                j = lambda_arr[j]
            i = lambda_arr[i]

    # Create lower triangular matrix by zeroing upper triangle
    L = np.tril(H)

    return L
