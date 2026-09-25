import numpy as np

import src.robot_models.parameters as p
from src.robot_models.planar_2dof_model import get_robot_matrices
def compute_control_input(q, dq):
    _, _, G = get_robot_matrices(q, dq)
    tau_nominal = 12.0 * (p.q_target - q) - 3.5 * dq + G
    return np.clip(tau_nominal, -p.TORQUE_LIMIT, p.TORQUE_LIMIT)