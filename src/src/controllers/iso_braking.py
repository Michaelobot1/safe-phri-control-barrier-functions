import numpy as np
import src.robot_models.parameters as p
from src.robot_models.planar_2dof_model import forward_kinematics, get_robot_matrices



# Persist the braking activation state across separate time integration loops
braking_triggered = False



def compute_control_input(q, dq, x_human):


    """
    Computes a state-dependent switching safety control torque input.
    If the end-effector enters the ISO protective stop zone, the tracking loop 
    is disconnected and a high-gain joint velocity dampening torque is applied
    to snap freeze the arm links mid-air.
    """

    
    global braking_triggered
    
    # 1. Calculate operational space kinematics
    _, x_ee = forward_kinematics(q)
    
    # 2. Measure physical proximity space separation distance
    dist_to_human = np.linalg.norm(x_ee - x_human)
    
    # Check if the protective stopping bubble threshold has been breached
    if dist_to_human <= p.D_stop:
        braking_triggered = True
        
    # 3. Apply state-dependent heuristic switching criteria (Eq. 32 of Manuscript)
    if braking_triggered:
        # High-Gain Velocity Dampening joint braking torque loop execution
        tau_applied = -15.0 * dq
    else:
        # Pass through regular nominal trajectory controller configurations
        _, _, G = get_robot_matrices(q, dq)
        tau_nominal = 12.0 * (p.q_target - q) - 3.5 * dq + G
        tau_applied = np.clip(tau_nominal, -p.TORQUE_LIMIT, p.TORQUE_LIMIT)
        
    return tau_applied

def reset_brake_state():
    """Resets the persistent internal global tracking variable before a new simulation execution."""
    global braking_triggered
    braking_triggered = False
