from dynamics.continuous_dynamics.HandC import HandC
from dynamics.continuous_dynamics.EnergyAndMomentum import EnergyAndMomentum
from dynamics.CMM import CMM


class ContinuousDynamics:
    """
    Class for computing and storing continuous dynamics quantities.

    Attributes:
        H_matrix: Joint-space inertia matrix
        C_terms: Coriolis, centrifugal, and gravity terms
        KE: Kinetic energy
        PE: Potential energy
        p_com: Center of mass position
        v_com: Center of mass velocity
        CAM: Centroidal angular momentum
        CMMat: Centroidal momentum matrix (Jacobian of CAM)
    """

    def __init__(self, sys):
        """
        Initialize ContinuousDynamics object.

        Parameters:
            sys: System containing model and state information
        """

        # Compute inertia matrix and bias terms
        H, C = HandC(self, sys)

        self.H_matrix = H
        self.C_terms = C

        # Compute energy and momentum quantities
        KE, PE, p_com, v_com, cam = EnergyAndMomentum(self, sys)

        self.KE = KE
        self.PE = PE
        self.p_com = p_com
        self.v_com = v_com
        self.CAM = cam

        # Compute centroidal momentum matrix
        A = CMM(self, sys)
        self.CMMat = A
