import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import src.robot_models.parameters as p
from src.robot_models.planar_2dof_model import forward_kinematics, get_robot_matrices, compute_safety_metrics
import src.controllers.naive_tracker as naive
import src.controllers.iso_braking as iso
import src.controllers.ecbf_qp_filter as cbf

os.makedirs('data_outputs', exist_ok=True)
os.makedirs('figures', exist_ok=True)

results_summary = {}

# ==============================================================================
# 1. RUN SIMULATIONS
# ==============================================================================
for mode in ['unconstrained', 'iso_braking', 'ecbf_qp']:
    q = p.q_start.copy()
    dq = np.zeros(2, dtype=float)
    iso.reset_brake_state()
    
    log_time, log_h, log_robot, log_human, log_solve = [], [], [], [], []
    
    for step in range(p.steps):
        t = step * p.dt
        
        if mode == 'unconstrained':
            x_human = np.array([0.55, 0.65 - 0.28 * t])
            v_human = np.array([0.0, -0.28])
            a_human = np.array([0.0, 0.0])
        else:
            x_human = np.array([0.45, 0.50 - 0.08 * t])
            v_human = np.array([0.0, -0.08])
            a_human = np.array([0.0, 0.0])
            
        h, h_dot, dh_dq = compute_safety_metrics(q, dq, x_human)
        x_elbow, x_ee = forward_kinematics(q)
        
        if mode == 'unconstrained':
            tau_applied = naive.compute_control_input(q, dq)
            solve_time_us = 0.0
        elif mode == 'iso_braking':
            tau_applied = iso.compute_control_input(q, dq, x_human)
            solve_time_us = 0.0
        elif mode == 'ecbf_qp':
            t0 = time.perf_counter()
            tau_applied = cbf.compute_control_input(q, dq, x_human, v_human, a_human)
            solve_time_us = (time.perf_counter() - t0) * 1e6
            
        M, C, G = get_robot_matrices(q, dq)
        ddq = np.linalg.inv(M) @ (tau_applied - C @ dq - G)
        dq += ddq * p.dt
        q += dq * p.dt
        
        if step % 8 == 0:
            log_time.append(t)
            log_h.append(h)
            log_robot.append((np.array([0.0, 0.0]), x_elbow.copy(), x_ee.copy()))
            log_human.append(x_human.copy())
            log_solve.append(solve_time_us)
            
    results_summary[mode] = {
        'time': log_time, 'h': log_h, 'robot': log_robot, 'human': log_human,
        'min_h': min(log_h),
        'accum_err': np.sum(np.linalg.norm(p.q_target - q, axis=0)) * p.dt,
        'avg_solve': np.mean(log_solve)
    }

# ==============================================================================
# 2. EXPORT CSV QUANTITATIVE METRICS
# ==============================================================================
csv_path = 'data_outputs/table2_quantitative_metrics.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Control Architecture Strategy', 'Minimum Safety h(t)', 'Accumulated Path Error', 'Avg Solve Speed (us)'])
    for m in ['unconstrained', 'iso_braking', 'ecbf_qp']:
        writer.writerow([
            m.upper(), 
            f"{results_summary[m]['min_h']:.4f}", 
            f"{results_summary[m]['accum_err']:.4f}", 
            f"{results_summary[m]['avg_solve']:.2f}"
        ])

color_map = {'unconstrained': '#dc2626', 'iso_braking': '#e67e22', 'ecbf_qp': '#16a34a'}
title_map = {
    'unconstrained': 'Unconstrained Baseline',
    'iso_braking': 'ISO 15066 Proximity Stop',
    'ecbf_qp': 'Proposed ECBF-QP Filter'
}

def draw_target(ax):
    ax.scatter([p.x_target[0]], [p.x_target[1]], color='#1e293b', marker='+', s=90, lw=1.8, zorder=5)
    ax.add_patch(plt.Circle(p.x_target, 0.03, color='#1e293b', fill=False, lw=1.2, zorder=5))

# ==============================================================================
# 3. GENERATE 4-ROW SINGLE-COLUMN COMBINED FIGURES (3 SNAPSHOTS + GRAPH)
# ==============================================================================
print("[PROCESSING] Generating 4-panel single-column stacked figures...")
for m in ['unconstrained', 'iso_braking', 'ecbf_qp']:
    # 4 rows, 1 col layout fitted for single IEEE column (~3.6 in wide x 11.0 in tall)
    fig, axes = plt.subplots(4, 1, figsize=(3.6, 11.0), dpi=300)
    
    robot_hist = results_summary[m]['robot']
    human_hist = results_summary[m]['human']
    time_hist = results_summary[m]['time']
    h_vals = results_summary[m]['h']
    
    min_idx = int(np.argmin(h_vals))
    phase_indices = [int(len(time_hist) * 0.10), min_idx, int(len(time_hist) * 0.90)]
    phase_titles = ['Phase 1: Approach', 'Phase 2: Closest Interaction', 'Phase 3: Completion']
    
    # Panels 1 to 3: Spatial Workspaces
    for ax, idx, p_title in zip(axes[:3], phase_indices, phase_titles):
        ax.set_xlim(-0.1, 1.0); ax.set_ylim(-0.1, 1.0)
        ax.set_aspect('equal'); ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_title(f"{p_title} (t = {time_hist[idx]:.2f}s)", fontsize=8, fontweight='bold')
        ax.set_xlabel("X (m)", fontsize=7); ax.set_ylabel("Y (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        
        base, elbow, ee = robot_hist[idx]
        hum = human_hist[idx]
        
        ax.plot([base[0], elbow[0], ee[0]], [base[1], elbow[1], ee[1]], 'o-', color=color_map[m], lw=2.5, markersize=5)
        ax.add_patch(plt.Circle(hum, p.R_safe, color='#ef4444', alpha=0.22, fill=True))
        ax.scatter(hum[0], hum[1], color='#8b5cf6', s=45, edgecolors='black', zorder=4)
        draw_target(ax)

    # Panel 4: Barrier Clearance Metric Plot h(t)
    ax_graph = axes[3]
    ax_graph.set_xlim(0, p.t_max); ax_graph.set_ylim(-0.05, 0.40)
    ax_graph.grid(True, linestyle=':', alpha=0.5)
    ax_graph.set_title("Barrier Clearance Margin $h(t)$", fontsize=8, fontweight='bold')
    ax_graph.set_xlabel("Time Horizon (s)", fontsize=7); ax_graph.set_ylabel("Margin $h$", fontsize=7)
    ax_graph.tick_params(labelsize=6)
    
    ax_graph.plot(time_hist, h_vals, color=color_map[m], lw=1.8, label="$h(t)$")
    ax_graph.axhline(0, color='#dc2626', linestyle='--', lw=1.2, label="Violation Limit ($h=0$)")
    ax_graph.legend(loc="upper right", fontsize=6)

    plt.suptitle(f"{title_map[m]}", fontsize=10, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"figures/fig_{m}_single_column_all_in_one.png", bbox_inches='tight')
    plt.close()

# ==============================================================================
# 4. GENERATE ANIMATED GIFS
# ==============================================================================
print("[PROCESSING] Exporting video animations (GIF)...")
for m in ['unconstrained', 'iso_braking', 'ecbf_qp']:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    
    robot_hist = results_summary[m]['robot']
    human_hist = results_summary[m]['human']
    time_hist = results_summary[m]['time']
    h_hist = results_summary[m]['h']
    
    def update(frame):
        ax1.clear(); ax2.clear()
        
        ax1.set_xlim(-0.1, 1.0); ax1.set_ylim(-0.1, 1.0)
        ax1.set_aspect('equal'); ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_title(f"{title_map[m]} (t = {time_hist[frame]:.2f}s)", fontsize=10, fontweight='bold')
        ax1.set_xlabel("X (m)", fontsize=9); ax1.set_ylabel("Y (m)", fontsize=9)
        
        base, elbow, ee = robot_hist[frame]
        hum = human_hist[frame]
        
        ee_path = np.array([r[2] for r in robot_hist[:frame+1]])
        if len(ee_path) > 0:
            ax1.plot(ee_path[:, 0], ee_path[:, 1], '--', color=color_map[m], alpha=0.5, lw=1.5)
            
        ax1.plot([base[0], elbow[0], ee[0]], [base[1], elbow[1], ee[1]], 'o-', color=color_map[m], lw=3.5, markersize=6)
        ax1.add_patch(plt.Circle(hum, p.R_safe, color='#ef4444', alpha=0.25, fill=True))
        ax1.scatter(hum[0], hum[1], color='#8b5cf6', s=70, edgecolors='black', zorder=4)
        draw_target(ax1)
        
        ax2.set_xlim(0, p.t_max); ax2.set_ylim(-0.05, 0.40); ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.set_title("Barrier Safety Margin $h(t)$", fontsize=10, fontweight='bold')
        ax2.set_xlabel("Time (s)", fontsize=9); ax2.set_ylabel("Clearance $h$", fontsize=9)
        ax2.plot(time_hist[:frame+1], h_hist[:frame+1], color=color_map[m], lw=2.2)
        ax2.axhline(0, color='#dc2626', linestyle='--', lw=1.5, label="Boundary $h=0$")
        ax2.legend(loc="upper right", fontsize=8)
        
        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(time_hist), interval=35)
    anim.save(f"figures/animation_{m}.gif", writer=PillowWriter(fps=30))
    plt.close()

print("\n[SUCCESS] Generated 4-panel single-column figures (figures/fig_*_single_column_all_in_one.png).")