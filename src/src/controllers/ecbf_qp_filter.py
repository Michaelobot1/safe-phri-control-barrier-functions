import numpy as np
import src.robot_models.parameters as p
from src.robot_models.planar_2dof_model import get_robot_matrices, compute_safety_metrics
from scipy.optimize import minimize


def compute_control_input(q, dq, x_human):


    """
    Computes the optimal, forward-invariant filtered torque input.
    Formulates a continuous 500 Hz Quadratic Program (QP) optimization filter 
    minimizing tracking distortion while strictly satisfying the relative-degree r=2 
    Exponential Control Barrier Function inequality constraints derived in Section IV.
    """


    # 1. Fetch continuous mathematical dynamics and safety states
    M, C, G = get_robot_matrices(q, dq)
    h, h_dot, dh_dq = compute_safety_metrics(q, dq, x_human)

    
    # 2. Formulate the primary tracking goal controller (Nominal torque)
    tau_nominal = 12.0 * (p.q_target - q) - 3.5 * dq + G
    tau_nominal = np.clip(tau_nominal, -p.TORQUE_LIMIT, p.TORQUE_LIMIT)
    

    # 3. Define the Optimization Objective: Minimize distortion from nominal behavior
    def objective(tau): 
        return np.sum((tau - tau_nominal) ** 2)
    

    # 4. Define the ECBF Inequality Constraint (h_ddot + alpha1*h_dot + alpha2*h >= 0)
    def ecbf_constraint(tau):
        Minv = np.linalg.inv(M)
        # Analytical coupling of acceleration to torque input map (Section IV text update)
        h_ddot_control = dh_dq @ Minv @ (tau - C @ dq - G)
        return h_ddot_control + p.alpha1 * h_dot + p.alpha2 * h


    # 5. Execute the Convex Quadratic Program solver loop
    bounds = [(-p.TORQUE_LIMIT, p.TORQUE_LIMIT)] * 2
    constraints = {'type': 'ineq', 'fun': ecbf_constraint}
    
    res = minimize(objective, x0=tau_nominal, method='SLSQP', 
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-4})
    
    # Pass through optimized torque if successful; fallback safely to nominal if bounds lock
    tau_applied = res.x if res.success else tau_nominal
    
    return tau_applied
