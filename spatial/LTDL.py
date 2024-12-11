import numpy as np


def LTDL(H, lambda_arr):
    """
    Factorize H -> L'*D*L exploiting branch-induced sparsity.

    Computes unit-lower-triangular matrix L and diagonal matrix D satisfying
    L'*D*L = H, where H is a symmetric, positive-definite matrix with 
    branch-induced sparsity pattern determined by lambda.

    Parameters:
        H: Symmetric positive-definite matrix (e.g., joint-space inertia matrix)
        lambda_arr: Parent array satisfying 0 <= lambda[i] < i for all i

    Returns:
        L: Unit-lower-triangular matrix
        D: Diagonal matrix
        Such that L.T @ D @ L = H
    """

    # Work on a copy of H to avoid modifying the input
    H = H.copy()
    n = H.shape[0]

    # Main factorization loop
    for k in range(n-1, -1, -1):
        i = lambda_arr[k]
        while i != 0:
            a = H[k, i] / H[k, k]
            j = i
            while j != 0:
                H[i, j] = H[i, j] - a * H[k, j]
                j = lambda_arr[j]
            H[k, i] = a
            i = lambda_arr[i]

    # Extract D and construct L
    D = np.diag(np.diag(H))
    L = np.eye(n)
    for i in range(1, n):
        L[i, 0:i] = H[i, 0:i]

    return L, D
