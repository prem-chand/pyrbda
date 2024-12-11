import numpy as np


def mpyLit(L, lambda_arr, x):
    """
    Multiply vector by inverse transpose of L factor from LTL or LTDL.

    Computes inv(L')*x where L is the lower-triangular matrix from either LTL or LTDL
    and lambda_arr is the parent array describing the sparsity pattern in L.

    Parameters:
        L: Lower-triangular matrix
        lambda_arr: Parent array describing sparsity pattern
        x: Input vector to multiply

    Returns:
        y: Result of inv(L')*x multiplication
    """
    n = L.shape[0]
    x = x.copy()  # Create copy to preserve input vector

    for i in range(n-1, -1, -1):  # Iterate from n-1 down to 0
        x[i] = x[i] / L[i, i]
        j = lambda_arr[i]
        while j != 0:
            x[j] = x[j] - L[i, j] * x[i]
            j = lambda_arr[j]

    return x
