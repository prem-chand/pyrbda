from typing import Tuple, Dict, Any
import numpy as np

from spatial.jcalc import jcalc
from spatial.crm import crm
from spatial.crf import crf
from spatial.get_gravity import get_gravity
from .CoriolisAndGravityTerms import compute_coriolis_gravity_terms
from spatial.InertiaMatrix import compute_inertia_matrix


def compute_dynamics_terms(system: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate coefficients of the joint-space equation of motion:
    tau = H(q)qdd + C(q, qd, f_ext)
    
    Args:
        system: System containing model and state information
        
    Returns:
        Tuple containing:
            - H: Joint-space inertia matrix
            - C: Vector of gravity, external-force, and velocity-product terms
    """
    model = system.Model
    q = system.qpos0
    qd = system.qvel0
    
    # Compute indices for different joint types
    v_indices, q_indices = _compute_joint_indices(model)
    
    # Initialize state variables
    state_vars = _initialize_state_variables(model, q, qd, v_indices, q_indices)
    
    # Forward pass computations
    state_vars = _forward_pass(system, state_vars, v_indices, q_indices)
    
    # Compute final terms
    C = compute_coriolis_gravity_terms(system, q, state_vars['S'], 
                                     state_vars['Xup'], state_vars['fvp'])
    H = compute_inertia_matrix(system, q, state_vars['S'], state_vars['Xup'])
    
    return H, C

def _compute_joint_indices(model: Any) -> Tuple[list, list]:
    """Compute indices for different joint types."""
    idx1 = slice(0, 6)  # free joint
    idx2 = [i+6 for i in range(12)]  # revolute joints
    v_indices = [idx1, *idx2]
    
    idx3 = [i+7 for i in range(13)]  # all joints
    q_indices = [slice(0, 7), *idx3]
    
    return v_indices, q_indices

def _initialize_state_variables(model: Any, q: np.ndarray, qd: np.ndarray, 
                              v_indices: list, q_indices: list) -> Dict:
    """Initialize state variables for dynamics computations."""
    return {
        'Xup': {},
        'v': {},
        'avp': {},
        'fvp': {},
        'S': {},
        'a_grav': get_gravity(model)
    }

def _forward_pass(system: Any, state_vars: Dict, v_indices: list, 
                 q_indices: list) -> Dict:
    """Perform forward pass computations for velocities and accelerations."""
    model = system.Model
    q = system.qpos0
    qd = system.qvel0
    
    for i in range(model.params.nb - 1):
        XJ, state_vars['S'][i] = jcalc(model.jtype[i], q[q_indices[i]])
        vJ = (state_vars['S'][i] @ qd[v_indices[i]]).reshape((-1, 1))
        
        state_vars['Xup'][i] = XJ @ model.Xtree[i]
        
        # Update velocities and accelerations
        _update_vel_acc(i, model, state_vars, vJ)
        
        # Compute force-velocity products
        state_vars['fvp'][i] = _compute_fvp(i, model, state_vars)
    
    return state_vars

def _update_vel_acc(i: int, model: Any, state_vars: Dict, vJ: np.ndarray) -> None:
    """Update velocities and accelerations for each joint."""
    if model.parent_id[i+1] == 0:
        state_vars['v'][i] = vJ
        state_vars['avp'][i] = state_vars['Xup'][i] @ (-state_vars['a_grav'])
    else:
        parent = model.parent_id[i+1]
        state_vars['v'][i] = state_vars['Xup'][i] @ state_vars['v'][parent] + vJ
        state_vars['avp'][i] = (state_vars['Xup'][i] @ state_vars['avp'][parent] + 
                               crm(state_vars['v'][i]) @ vJ)

def _compute_fvp(i: int, model: Any, state_vars: Dict) -> np.ndarray:
    """Compute force-velocity products."""
    return (model.fullinertia[i+1] @ state_vars['avp'][i] + 
            crf(state_vars['v'][i]) @ model.fullinertia[i+1] @ state_vars['v'][i])
