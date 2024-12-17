from dynamics.continuous_dynamics.HandC import HandC
from dynamics.continuous_dynamics.EnergyAndMomentum import EnergyAndMomentum
from dynamics.CMM import CMM


class ContinuousDynamics:
    """
    Class for computing and storing continuous dynamics quantities.

    Attributes:
        H_matrix (np.ndarray): Joint-space inertia matrix
        C_terms (np.ndarray): Coriolis, centrifugal, and gravity terms
        KE (float): Kinetic energy
        PE (float): Potential energy
        p_com (np.ndarray): Center of mass position
        v_com (np.ndarray): Center of mass velocity
        CAM (np.ndarray): Centroidal angular momentum
        CMMat (np.ndarray): Centroidal momentum matrix (Jacobian of CAM)
    """

    def __init__(self, sys: 'System'):
        """
        Initialize ContinuousDynamics object.

        Parameters:
            sys (System): System containing model and state information
        """
        # Compute inertia matrix and bias terms
        H, C = HandC(sys)

        self.H_matrix = H
        self.C_terms = C

        # Compute energy and momentum quantities
        KE, PE, p_com, v_com, cam = EnergyAndMomentum(sys)

        self.KE = KE
        self.PE = PE
        self.p_com = p_com
        self.v_com = v_com
        self.CAM = cam

        # Compute centroidal momentum matrix
        A = CMM(self, sys)
        self.CMMat = A
