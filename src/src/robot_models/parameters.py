import numpy as np


# PHYSICAL ROBOT SPECIFICATIONS (Table I of Manuscript Layout)

l1, l2 = 0.5, 0.4    # Link lengths of the 2-DOF industrial manipulator (meters)
m1, m2 = 2.5, 1.8    # Link masses (kg)
g = 9.81             # Acceleration due to gravity constants (m/s^2)


# SAFETY BOUNDARIES & ENVELOPE DESIGN

R_safe = 0.15        # Invisible protective safety bubble radius around human (meters)
D_stop = 0.25        # Legacy ISO 15066 Proximity stopping trigger entry threshold (meters)


# EXPONENTIAL CONTROL BARRIER FUNCTION TUNING GAINS
# Critically overdamped pole placement criteria satisfied: alpha2 >= 2*sqrt(alpha1)

alpha1 = 8.0
alpha2 = 20.0


# SIMULATION RESOLUTION & HARD ACTUATOR BOUNDS

dt = 0.002           # High-frequency loop resolution to stop numerical chatter (500 Hz)
t_max = 5.0          # Evaluation time horizon length (seconds)
steps = int(t_max / dt)
TORQUE_LIMIT = 40.0  # Absolute maximum permitted actuator motor torque saturation (Nm)


# TRANSVERSE OPERATIONAL TRACKING STATES

q_start = np.array([0.1, 0.3], dtype=float)   # Uniform initial joint state angles (radians)
q_target = np.array([1.2, -0.4], dtype=float) # Intended nominal configuration target goal
x_target = np.array([0.70, 0.22])             # Spatial Cartesian destination point dot coordinate
