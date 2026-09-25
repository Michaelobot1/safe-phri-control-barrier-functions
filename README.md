```markdown
# Control Barrier Functions for Safe Physical Human-Robot Interaction: A Comparative Review and Simulation-Based Evaluation

Official code repository and benchmark evaluation suite for the paper **"Control Barrier Functions for Safe Physical Human-Robot Interaction: A Comparative Review and Simulation-Based Evaluation"** by Obot Michael O.

This workspace provides a simulation and comparative analysis framework evaluating Control Barrier Function (CBF) formulations—including Exponential CBFs (ECBFs), robust variants under unmodeled human trajectories, and classical ISO 15066 Speed and Separation Monitoring (SSM)—for torque-controlled robotic manipulators in shared human workspaces.

---

## Abstract Summary

To ensure that physical human-robot interaction (pHRI) remains strictly safe, this work evaluates optimization-based safety filters designed to enforce hard distance constraints during close-proximity interactions. Because human-robot distance constraints exhibit a high relative degree with respect to joint torque inputs, standard control-affine safety filters are transposed into higher relative degree formulations. 

This repository benchmarks:
* **Classical Proximity Mechanisms:** ISO 15066 Speed and Separation Monitoring (SSM) emergency braking.
* **Unconstrained Tracking:** Nominal PD inverse dynamics tracking baseline.
* **Higher-Order Optimization Filters:** Exponential Control Barrier Functions (ECBF-QP) mapped through convex Quadratic Programs.

Evaluations quantify safety margin preservation $h(t)$, optimization solve times ($\mu\text{s}$), and task tracking conservatism.

---

## Repository Architecture


```

text
.
├── data_outputs/
│   └── table2_quantitative_metrics.csv       # Minimum h(t), path error, solve speeds
├── figures/
│   ├── animation_ecbf_qp.gif                 # Animated workspace trajectory & clearance
│   ├── animation_iso_braking.gif
│   ├── animation_unconstrained.gif
│   ├── fig_ecbf_qp_single_column_all_in_one.png  # 4-panel IEEE single-column figure
│   ├── fig_iso_braking_single_column_all_in_one.png
│   └── fig_unconstrained_single_column_all_in_one.png
└── src/
├── **init**.py
├── evaluate_benchmarks.py                # Comparative benchmark execution script
├── controllers/
│   ├── **init**.py
│   ├── ecbf_qp_filter.py                 # Active ECBF-QP optimization filter
│   ├── iso_braking.py                    # ISO 15066 proximity-braking baseline
│   └── naive_tracker.py                  # Nominal inverse dynamics tracker
└── robot_models/
├── **init**.py
├── parameters.py                     # Kinematic, dynamic, and safety bounds
└── planar_2dof_model.py              # FK, Jacobian, dynamics, and barrier h(t)

```

---

## Mathematical Formulation

### 1. Manipulator System Dynamics
The 2-DOF planar manipulator dynamics are formulated as:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = \tau$$

Where $q \in \mathbb{R}^2$ represents joint positions, $M(q) \in \mathbb{R}^{2 \times 2}$ is the mass matrix, $C(q,\dot{q}) \in \mathbb{R}^{2 \times 2}$ represents Coriolis forces, $G(q) \in \mathbb{R}^2$ is gravity, and $\tau \in \mathbb{R}^2$ is the control torque input vector.

### 2. Relative-Degree Two Safety Barrier Constraint
Safety is defined via the task-space separation distance between the robot end-effector $p_{\text{ee}}(q) \in \mathbb{R}^2$ and a time-varying human operator trajectory $p_h(t) \in \mathbb{R}^2$:

$$h(q, t) = \Vert{}p_{\text{ee}}(q) - p_h(t)\Vert{}^2 - R_{\text{safe}}^2 \ge 0$$

Because control input $\tau$ appears at the acceleration level ($\ddot{q}$), $h(q,t)$ exhibits **relative degree 2**. The second time derivative incorporates dynamic human velocity ($\dot{p}_h$) and acceleration ($\ddot{p}_h$):

$$\ddot{h}(q, \dot{q}, \ddot{q}, t) = J_h(q) \ddot{q} + \dot{J}_h(q, \dot{q})\dot{q} - 2(p_{\text{ee}} - p_h)^T \ddot{p}_h + 2\Vert{}\dot{p}_{\text{ee}} - \dot{p}_h\Vert{}^2$$

### 3. Convex Optimization (ECBF-QP) Filter
The real-time safety filter solves for joint acceleration $\ddot{q}_{\text{safe}}$:

$$\min_{\ddot{q}} \frac{1}{2} \Vert{}\ddot{q} - \ddot{q}_{\text{nom}}\Vert{}^2$$
$$\text{subject to } \quad \ddot{h}(q, \dot{q}, \ddot{q}, t) + \alpha_1 \dot{h}(q, \dot{q}, t) + \alpha_2 h(q, t) \ge 0$$

The resulting safe joint acceleration is mapped back to joint torques via inverse dynamics:

$$\tau_{\text{applied}} = M(q)\ddot{q}_{\text{safe}} + C(q, \dot{q})\dot{q} + G(q)$$

---

## Installation & Setup

1. **Clone the repository:**

```

bash
git clone [https://github.com/your-username/cbf-phri-comparative-review.git](https://github.com/your-username/cbf-phri-comparative-review.git?utm_source=gemini)
cd cbf-phri-comparative-review

```

2. **Install dependencies:**

```

bash
pip install numpy scipy matplotlib pillow

```

---

## Execution & Benchmarking

Run the comparative evaluation runner module from the root directory:


```

bash
python -m src.evaluate_benchmarks

```

### Exported Artifacts
* **Quantitative Data (`data_outputs/table2_quantitative_metrics.csv`):** Comparative metrics on minimum safety clearance $\min h(t)$, accumulated tracking error, and mean QP solve time ($\mu\text{s}$).
* **IEEE Publication Visualizations (`figures/fig_*_single_column_all_in_one.png`):** Stacked 4-panel figures showing interaction Phase 1 (Approach), Phase 2 (Closest Distance), Phase 3 (Completion), and the clearance trajectory $h(t)$.
* **Workspace Animations (`figures/animation_*.gif`):** Side-by-side workspace and barrier clearance dynamics.

---

## IEEE LaTeX Figure Embedding

To embed the composite single-column figure into an IEEE double-column template:


```

latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{figures/fig_ecbf_qp_single_column_all_in_one.png}
\caption{Spatiotemporal evolution and safety margin preservation $h(t)$ of the ECBF-QP filter across interaction phases.}
\label{fig:ecbf_qp_results}
\end{figure}

```

---

## Citation

If you use this benchmark suite or control implementation in your research, please cite:


```

bibtex
@article{obot2026cbf,
author    = {Obot, Michael O.},
title     = {Control Barrier Functions for Safe Physical Human-Robot Interaction: A Comparative Review and Simulation-Based Evaluation},
journal   = {IEEE Transactions on Robotics},
year      = {2026}
}

```


```