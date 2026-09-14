import numpy as np
import src.robot_models.parameters as p
from src.robot_models.planar_2dof_model import get_robot_matrices


def compute_control_input(q, dq):


    """
    Computes a standard Proportional-Derivative (PD) tracking torque input
    with non-linear Gravity Compensation. This method completely ignores workspace 
    obstacles, serving as the paper's negative baseline tracking control model.
    """

    
    # Evaluate current physical rigid robot dynamics parameters
    _, _, G = get_robot_matrices(q, dq)
    
    # Joint-Space Nominal Trajectory Tracking Torque Equation
    tau_nominal = 12.0 * (p.q_target - q) - 3.5 * dq + G
    
    # Enforce strict physical actuator saturation boundaries
    tau_applied = np.clip(tau_nominal, -p.TORQUE_LIMIT, p.TORQUE_LIMIT)
    
    return tau_applied
