import numpy as np

l1, l2 = 0.5, 0.4    # Link lengths (meters)
m1, m2 = 2.5, 1.8    # Link masses (kg)
g = 9.81             # Gravity (m/s^2)

R_safe = 0.15        # Safety bubble radius (meters)
D_stop = 0.25        # ISO 15066 stopping threshold (meters)

alpha1 = 8.0         # ECBF tuning gain 1
alpha2 = 20.0        # ECBF tuning gain 2

dt = 0.002           # 500 Hz resolution
t_max = 5.0          # Time horizon (seconds)
steps = int(t_max / dt)
TORQUE_LIMIT = 40.0  # Max actuator torque (Nm)

q_start = np.array([0.1, 0.3], dtype=float)   # Initial joint angles
q_target = np.array([1.2, -0.4], dtype=float) # Target joint configuration
x_target = np.array([0.70, 0.22])             # Target Cartesian point