import numpy as np

import src.robot_models.parameters as p
from src.robot_models.planar_2dof_model import forward_kinematics, get_robot_matrices, compute_safety_metrics

from scipy.optimize import minimize

def compute_control_input(q, dq, x_human, v_human=np.zeros(2), a_human=np.zeros(2)):
    M, C, G = get_robot_matrices(q, dq)
    h, h_dot, dh_dq = compute_safety_metrics(q, dq, x_human)
    
    q1, q2 = float(q[0]), float(q[1])
    J = np.array([[-p.l1*np.sin(q1) - p.l2*np.sin(q1+q2), -p.l2*np.sin(q1+q2)],
                  [ p.l1*np.cos(q1) + p.l2*np.cos(q1+q2),  p.l2*np.cos(q1+q2)]])
    v_ee = J @ dq
    
    tau_nominal = 12.0 * (p.q_target - q) - 3.5 * dq + G
    tau_nominal = np.clip(tau_nominal, -p.TORQUE_LIMIT, p.TORQUE_LIMIT)
    
    def objective(tau): 
        return np.sum((tau - tau_nominal) ** 2)
    
    def ecbf_constraint(tau):
        Minv = np.linalg.inv(M)
        ddq = Minv @ (tau - C @ dq - G)
        _, x_ee = forward_kinematics(q)
        dist_vec = x_ee - x_human
        rel_vel = v_ee - v_human
        
        h_ddot = 2.0 * (np.dot(rel_vel, rel_vel) + np.dot(dist_vec, J @ ddq - a_human))
        return h_ddot + p.alpha1 * h_dot + p.alpha2 * h

    bounds = [(-p.TORQUE_LIMIT, p.TORQUE_LIMIT)] * 2
    constraints = {'type': 'ineq', 'fun': ecbf_constraint}
    
    res = minimize(objective, x0=tau_nominal, method='SLSQP', 
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-4})
    
    return res.x if res.success else tau_nominal