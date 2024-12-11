"""
   Spatial/planar cross-product operator (force).

   Calculates the 6x6 (or 3x3) matrix such that crf(v)*f is the cross product
   of the motion vector v with the force vector f.

   Parameters:
       v: Motion vector (6D for spatial, 3D for planar)

   Returns:
       vcross: Cross-product operator matrix (6x6 for spatial, 3x3 for planar)
   """

import numpy as np
from spatial.crm import crm


def crf(v):

    vcross = -crm(v).T
    return vcross
