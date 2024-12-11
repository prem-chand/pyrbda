import numpy as np


def expandLambda(lambda_arr, nf):
    """
    Expand a parent array for use in sparse factorization algorithms.

    Calculates the expanded parent array from a given parent array and an array
    of joint motion freedoms. nf[i] is the degree of motion freedom allowed by joint i.

    Parameters:
        lambda_arr: Original parent array
        nf: Array of joint motion freedoms

    Returns:
        newLambda: Expanded parent array
    """

    N = len(lambda_arr)
    n = sum(nf)

    newLambda = np.arange(n)

    # Initialize map array
    map_arr = np.zeros(N + 1, dtype=int)

    for i in range(N):
        map_arr[i + 1] = map_arr[i] + nf[i]

    for i in range(N):
        newLambda[map_arr[i]:map_arr[i+1]] = map_arr[lambda_arr[i]]

    return newLambda
