import numpy as np


def mpyL(L, lambda_arr, x):
    """
    Multiply vector by L factor from LTL or LTDL.

    Computes L*x where L is the lower-triangular matrix from either LTL or LTDL
    and lambda_arr is the parent array describing the sparsity pattern in L.

    Parameters:
        L: Lower-triangular matrix
        lambda_arr: Parent array describing sparsity pattern
        x: Input vector to multiply

    Returns:
        y: Result of L*x multiplication
    """
    n = L.shape[0]
    y = x.copy()  # Create copy to preserve input vector

    for i in range(n):
        y[i] = L[i, i] * x[i]
        j = lambda_arr[i]
        while j != 0:
            y[i] = y[i] + L[i, j] * x[j]
            j = lambda_arr[j]

    return y
