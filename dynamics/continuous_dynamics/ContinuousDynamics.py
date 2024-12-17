from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from .HandC import compute_dynamics_terms
from .EnergyAndMomentum import EnergyAndMomentum
from ..CMM import CMM

@dataclass
class ContinuousDynamics:
    """
    Class for computing and storing continuous dynamics quantities.
    
    Attributes:
        H_matrix: Joint-space inertia matrix
        C_terms: Coriolis, centrifugal, and gravity terms
        kinetic_energy: System kinetic energy
        potential_energy: System potential energy
        com_position: Center of mass position
        com_velocity: Center of mass velocity
        centroidal_momentum: Centroidal angular momentum
        centroidal_momentum_matrix: Centroidal momentum matrix (Jacobian of CAM)
    """
    H_matrix: np.ndarray
    C_terms: np.ndarray
    kinetic_energy: float
    potential_energy: float
    com_position: np.ndarray
    com_velocity: np.ndarray
    centroidal_momentum: np.ndarray
    centroidal_momentum_matrix: Optional[np.ndarray] = None

    @classmethod
    def from_system(cls, system: Any) -> 'ContinuousDynamics':
        """
        Create ContinuousDynamics instance from a system.
        
        Args:
            system: System containing model and state information
            
        Returns:
            ContinuousDynamics instance with computed quantities
        """
        # Compute inertia matrix and bias terms
        H, C = compute_dynamics_terms(system)
        
        # Compute energy and momentum quantities
        KE, PE, p_com, v_com, cam = compute_energy_and_momentum(system)
        
        # Compute centroidal momentum matrix
        A = CMM(system)
        
        return cls(
            H_matrix=H,
            C_terms=C,
            kinetic_energy=KE,
            potential_energy=PE,
            com_position=p_com,
            com_velocity=v_com,
            centroidal_momentum=cam,
            centroidal_momentum_matrix=A
        )
