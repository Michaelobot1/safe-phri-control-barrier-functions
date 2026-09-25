import numpy as np
import src.robot_models.parameters as p

def forward_kinematics(q):
    q1, q2 = float(q[0]), float(q[1])
    x1 = p.l1 * np.cos(q1)
    y1 = p.l1 * np.sin(q1)
    x2 = x1 + p.l2 * np.cos(q1 + q2)
    y2 = y1 + p.l2 * np.sin(q1 + q2)
    return np.array([x1, y1]), np.array([x2, y2])

def get_robot_matrices(q, dq):
    q1, q2 = float(q[0]), float(q[1])
    dq1, dq2 = float(dq[0]), float(dq[1])

    M11 = p.m1*(p.l1/2)**2 + p.m2*(p.l1**2 + (p.l2/2)**2 + 2*p.l1*(p.l2/2)*np.cos(q2))
    M12 = p.m2*((p.l2/2)**2 + p.l1*(p.l2/2)*np.cos(q2))
    M = np.array([[M11, M12], [M12, p.m2*(p.l2/2)**2]])

    h_param = -p.m2*p.l1*(p.l2/2)*np.sin(q2)
    C = np.array([[h_param*dq2, h_param*(dq1 + dq2)], 
                  [-h_param*dq1, 0.0]])

    G1 = (p.m1*(p.l1/2) + p.m2*p.l1)*p.g*np.cos(q1) + p.m2*(p.l2/2)*p.g*np.cos(q1 + q2)
    G2 = p.m2*(p.l2/2)*p.g*np.cos(q1 + q2)
    G = np.array([G1, G2])
    
    return M, C, G

def compute_safety_metrics(q, dq, x_human):
    _, x_ee = forward_kinematics(q)
    dist_vec = x_ee - x_human
    dist = np.linalg.norm(dist_vec)
    
    h = dist**2 - p.R_safe**2
    
    q1, q2 = float(q[0]), float(q[1])
    J = np.array([[-p.l1*np.sin(q1) - p.l2*np.sin(q1+q2), -p.l2*np.sin(q1+q2)],
                  [ p.l1*np.cos(q1) + p.l2*np.cos(q1+q2),  p.l2*np.cos(q1+q2)]])
    
    v_ee = J @ dq
    h_dot = 2 * dist_vec.T @ v_ee
    dh_dq = 2 * dist_vec.T @ J
    
    return h, h_dot, dh_dq