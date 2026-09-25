import numpy as np

import src.robot_models.parameters as p
from src.robot_models.planar_2dof_model import forward_kinematics, get_robot_matrices

braking_triggered = False

def compute_control_input(q, dq, x_human):
    global braking_triggered
    _, x_ee = forward_kinematics(q)
    
    if np.linalg.norm(x_ee - x_human) <= p.D_stop:
        braking_triggered = True
        
    if braking_triggered:
        tau_applied = -15.0 * dq
    else:
        _, _, G = get_robot_matrices(q, dq)
        tau_nominal = 12.0 * (p.q_target - q) - 3.5 * dq + G
        tau_applied = np.clip(tau_nominal, -p.TORQUE_LIMIT, p.TORQUE_LIMIT)
        
    return tau_applied

def reset_brake_state():
    global braking_triggered
    braking_triggered = False