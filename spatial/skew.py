import numpy as np


def skew(in_array):
    """
    skew  convert 3D vector <--> 3x3 skew-symmetric matrix
    S=skew(v) and v=skew(A) calculate the 3x3 skew-symmetric matrix S
    corresponding to the given 3D vector v, and the 3D vector corresponding
    to the skew-symmetric component of the given arbitrary 3x3 matrix A. For
    vectors a and b, skew(a)*b is the cross product of a and b. If the
    argument is a 3x3 matrix then it is assumed to be A, otherwise it is
    assumed to be v. skew(A) produces a column-vector result, but skew(v)
    will accept a row or column vector argument.
    """
    in_array = np.asarray(in_array)

    if in_array.shape == (3, 3):  # do v = skew(A)
        out = 0.5 * np.array([in_array[2, 1] - in_array[1, 2],
                              in_array[0, 2] - in_array[2, 0],
                              in_array[1, 0] - in_array[0, 1]])
    else:  # do S = skew(v)
        out = np.array([[0, -in_array[2], in_array[1]],
                        [in_array[2], 0, -in_array[0]],
                        [-in_array[1], in_array[0], 0]])

    return out
