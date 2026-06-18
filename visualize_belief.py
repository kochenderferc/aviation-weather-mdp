"""
Belief update visualizations for the aviation weather POMDP.

Generates:
  plots/belief_update_example.png  — step-by-step Bayes update for one observation
  plots/belief_trajectory_*.png    — belief evolving across a full simulated flight
  plots/belief_simplex.png         — 2D simplex showing all sampled belief points
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from scenario import WEATHER_CONDITIONS, WAYPOINT_NAMES, WEATHER_TRANSITION
from pomdp import OBS_MODEL, OBS_LABELS, N_OBS, predict_belief, belief_update
from pomdp_solver import solve_pomdp, simulate_pomdp, get_action

WEATHER_COLORS = ['#2ecc71', '#f39c12', '#e74c3c']   # good, marginal, bad
WEATHER_LABELS = ['Good', 'Marginal', 'Bad']

# ─────────────────────────────────────────────────────────────
# 1. Step-by-step Bayes update
# ─────────────────────────────────────────────────────────────

def plot_belief_update_steps():
    """Show how a single observation reshapes the belief distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Bayes Belief Update — One Step\n"
                 "Prior: uniform (0.33 / 0.33 / 0.33)  |  Observation: bad / high-conf",
                 fontsize=12)

    prior = np.ones(3) / 3
    obs_idx = 5   # bad / high-conf

    # Panel 1: prior
    axes[0].bar(WEATHER_LABELS, prior, color=WEATHER_COLORS, edgecolor='k', linewidth=0.8)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Prior  b(w)")
    axes[0].set_ylabel("Probability")
    for i, v in enumerate(prior):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=11)

    # Panel 2: likelihood P(obs | w)
    likelihood = OBS_MODEL[:, obs_idx]
    axes[1].bar(WEATHER_LABELS, likelihood, color=WEATHER_COLORS, edgecolor='k', linewidth=0.8)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Likelihood  P('{OBS_LABELS[obs_idx]}' | w)")
    for i, v in enumerate(likelihood):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=11)

    # Panel 3: posterior
    unnorm = likelihood * prior
    posterior = unnorm / unnorm.sum()
    axes[2].bar(WEATHER_LABELS, posterior, color=WEATHER_COLORS, edgecolor='k', linewidth=0.8)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Posterior  b'(w)  ∝  P(obs|w)·b(w)")
    for i, v in enumerate(posterior):
        axes[2].text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=11)

    # Annotate the math
    fig.text(0.5, 0.02,
             "b'(w) = P(obs | w) · b(w) / Σ_{w'} P(obs | w') · b(w')",
             ha='center', fontsize=10, style='italic', color='#444')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("plots/belief_update_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/belief_update_example.png")


# ─────────────────────────────────────────────────────────────
# 2. Belief trajectory across a flight
# ─────────────────────────────────────────────────────────────

def plot_belief_trajectory(trajectory, label):
    """Stacked bar chart of belief at each waypoint visited."""
    # Filter to in-flight waypoints only
    steps = [s for s in trajectory if s['belief'] is not None]
    if not steps:
        return

    waypoints = [s['waypoint'] for s in steps]
    beliefs = np.array([s['belief'] for s in steps])   # shape (n_steps, 3)
    actions = [s['action'] for s in steps]
    obs = [s['obs'] for s in steps]
    true_wx = [s['true_weather'] for s in steps]

    n = len(steps)
    x = np.arange(n)
    width = 0.55

    fig, (ax_belief, ax_q) = plt.subplots(2, 1, figsize=(10, 7),
                                           gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle(f"POMDP Flight Simulation — True starting weather: {label}",
                 fontsize=13, fontweight='bold')

    # ── Belief bars ──
    bottoms = np.zeros(n)
    for i in range(3):
        ax_belief.bar(x, beliefs[:, i], width, bottom=bottoms,
                      color=WEATHER_COLORS[i], label=WEATHER_LABELS[i],
                      edgecolor='k', linewidth=0.6)
        for j in range(n):
            v = beliefs[j, i]
            if v > 0.07:
                ax_belief.text(j, bottoms[j] + v / 2, f"{v:.2f}",
                               ha='center', va='center', fontsize=9, color='white',
                               fontweight='bold')
        bottoms += beliefs[:, i]

    # Mark actions
    for j, (act, wp) in enumerate(zip(actions, waypoints)):
        color = '#c0392b' if act == 'divert' else '#27ae60'
        ax_belief.axvline(j, color=color, linestyle='--', linewidth=1.2, alpha=0.5)

    ax_belief.set_ylim(0, 1.05)
    ax_belief.set_ylabel("Belief  b(weather)", fontsize=11)
    ax_belief.set_xticks(x)
    ax_belief.set_xticklabels([
        f"{wp}\nobs: {o}\ntrue: {tw}\n→ {act}"
        for wp, o, tw, act in zip(waypoints, obs, true_wx, actions)
    ], fontsize=8)
    ax_belief.legend(loc='upper right', fontsize=9)
    ax_belief.set_title("Belief distribution at each waypoint")

    # ── Q-values ──
    q_cont = [s['q_continue'] for s in steps]
    q_div  = [float(s['q_divert']) for s in steps]
    ax_q.plot(x, q_cont, 'o-', color='#2980b9', label='Q(continue)', linewidth=2)
    ax_q.axhline(-20, color='#c0392b', linestyle='--', linewidth=1.5, label='Q(divert) = −20')
    ax_q.fill_between(x, q_cont, -20,
                       where=[qc >= -20 for qc in q_cont],
                       alpha=0.15, color='#2980b9')
    ax_q.set_ylabel("Q-value", fontsize=11)
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(waypoints, fontsize=9)
    ax_q.legend(fontsize=9)
    ax_q.set_title("Continue vs. Divert Q-values")
    ax_q.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"plots/belief_trajectory_{label.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────
# 3. Simplex plot — all-observations belief update fan
# ─────────────────────────────────────────────────────────────

def _to_2d(b):
    """Map 3-simplex belief to 2D equilateral triangle."""
    x = 0.5 * (2 * b[1] + b[2]) / (b[0] + b[1] + b[2])
    y = (np.sqrt(3) / 2) * b[2] / (b[0] + b[1] + b[2])
    return x, y


def plot_simplex_belief_fan():
    """Show how each possible observation shifts a uniform prior on the simplex."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal')
    ax.axis('off')
    fig.suptitle("Belief Simplex — How Each Observation Shifts the Prior\n"
                 "Prior = uniform (centre).  Arrows show posterior for each obs.",
                 fontsize=11)

    # Triangle vertices: good=left, marginal=right, bad=top
    verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    tri = plt.Polygon(verts, fill=False, edgecolor='#333', linewidth=2)
    ax.add_patch(tri)
    offset = 0.05
    ax.text(-offset, -offset, 'Good', ha='center', fontsize=11, color=WEATHER_COLORS[0], fontweight='bold')
    ax.text(1 + offset, -offset, 'Marginal', ha='center', fontsize=11, color=WEATHER_COLORS[1], fontweight='bold')
    ax.text(0.5, np.sqrt(3) / 2 + offset, 'Bad', ha='center', fontsize=11, color=WEATHER_COLORS[2], fontweight='bold')

    prior = np.ones(3) / 3
    px, py = _to_2d(prior)
    ax.plot(px, py, 'ko', markersize=10, zorder=5, label='Prior (uniform)')
    ax.text(px + 0.03, py, 'prior', fontsize=9)

    obs_colors = ['#1abc9c', '#16a085', '#e67e22', '#d35400', '#e74c3c', '#c0392b']
    for obs_idx, (color, obs_label) in enumerate(zip(obs_colors, OBS_LABELS)):
        unnorm = OBS_MODEL[:, obs_idx] * prior
        norm = unnorm.sum()
        if norm < 1e-12:
            continue
        posterior = unnorm / norm
        qx, qy = _to_2d(posterior)
        ax.annotate('', xy=(qx, qy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0))
        ax.plot(qx, qy, 'o', color=color, markersize=7, zorder=5)
        ax.text(qx + 0.02, qy + 0.01, obs_label.replace(' / ', '\n'),
                fontsize=7.5, color=color)

    ax.set_xlim(-0.2, 1.25)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(loc='lower center', fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/belief_simplex.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/belief_simplex.png")


# ─────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('plots', exist_ok=True)

    print("Solving POMDP...")
    gamma_per_wp = solve_pomdp()

    print("\nGenerating belief visualizations...")
    plot_belief_update_steps()
    plot_simplex_belief_fan()

    for wx_idx, wx_label in enumerate(['good', 'marginal', 'bad']):
        traj = simulate_pomdp(gamma_per_wp, initial_weather=wx_idx, seed=42)
        plot_belief_trajectory(traj, wx_label)

    print("\nAll belief plots saved.")
