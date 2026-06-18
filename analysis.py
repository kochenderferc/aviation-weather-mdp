"""
Two analyses that deepen the MDP/POMDP project:

  1. Sensitivity analysis — how do policies change as reward parameters vary?
     - Sweep divert penalty from −5 to −100
     - Sweep bad-weather continue cost from −20 to −150
     - Show policy flip boundaries

  2. Belief threshold visualization — where on the 2D simplex does the
     POMDP policy switch from continue → divert at each waypoint?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

WEATHER_COLORS = ['#2ecc71', '#f39c12', '#e74c3c']
WEATHER_LABELS = ['Good', 'Marginal', 'Bad']

def _to_2d(b):
    """Map 3-simplex belief to 2D equilateral triangle coords."""
    b = np.asarray(b, dtype=float)
    b = b / b.sum()
    x = 0.5 * (2 * b[1] + b[2])
    y = (np.sqrt(3) / 2) * b[2]
    return x, y

def _from_2d(x, y):
    """Inverse: 2D triangle coords → belief (good, marginal, bad)."""
    bad  = y / (np.sqrt(3) / 2)
    marg = x - y / np.sqrt(3)
    good = 1.0 - marg - bad
    return np.array([good, marg, bad])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def _run_value_iteration(reward_arrive, reward_divert, reward_bad,
                          reward_good=2.0, reward_marginal=-5.0,
                          gamma=0.95, tol=1e-6, max_iter=200):
    """
    Minimal self-contained value iteration for the MDP.
    Returns policy dict: state → 'continue' or 'divert'
    """
    from scenario import WAYPOINT_NAMES, N_WAYPOINTS, WEATHER_CONDITIONS, N_WEATHER, WEATHER_TRANSITION

    weather_rewards = [reward_good, reward_marginal, reward_bad]
    states = [(wp, w) for wp in range(N_WAYPOINTS) for w in range(N_WEATHER)]
    U = {s: 0.0 for s in states}
    U['arrived']  = reward_arrive
    U['diverted'] = reward_divert

    for _ in range(max_iter):
        delta = 0.0
        new_U = dict(U)
        for (wp, w) in states:
            # Q(continue)
            if wp < N_WAYPOINTS - 1:
                next_states = [(wp + 1, w2) for w2 in range(N_WEATHER)]
                q_cont = weather_rewards[w] + gamma * sum(
                    WEATHER_TRANSITION[w, w2] * U[(wp + 1, w2)]
                    for w2 in range(N_WEATHER)
                )
            else:  # last waypoint → arrive
                q_cont = weather_rewards[w] + reward_arrive
            # Q(divert)
            q_div = reward_divert
            new_U[(wp, w)] = max(q_cont, q_div)
            delta = max(delta, abs(new_U[(wp, w)] - U[(wp, w)]))
        U = new_U
        if delta < tol:
            break

    policy = {}
    for (wp, w) in states:
        if wp < N_WAYPOINTS - 1:
            q_cont = weather_rewards[w] + gamma * sum(
                WEATHER_TRANSITION[w, w2] * U[(wp + 1, w2)]
                for w2 in range(N_WEATHER)
            )
        else:
            q_cont = weather_rewards[w] + reward_arrive
        q_div = reward_divert
        policy[(wp, w)] = 'continue' if q_cont >= q_div else 'divert'
    return policy


def sensitivity_divert_penalty():
    """
    Sweep the divert penalty from −5 to −100.
    Show which states flip their policy.
    """
    from scenario import WAYPOINT_NAMES, N_WAYPOINTS, WEATHER_CONDITIONS

    penalties = np.linspace(-5, -100, 80)
    # Track actions for each (waypoint, weather) across penalty values
    # Focus on the three states that are interesting: wp×bad, KTRM×marginal
    states_of_interest = [
        (0, 2, 'KPHX, bad'),
        (1, 2, 'KBXK, bad'),
        (2, 2, 'KTRM, bad'),
        (2, 1, 'KTRM, marginal'),
        (3, 2, 'KSAN, bad'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sensitivity Analysis — Divert Penalty", fontsize=13, fontweight='bold')

    # Left: heatmap of policy across all states
    ax = axes[0]
    all_states = [(wp, w) for wp in range(N_WAYPOINTS) for w in range(3)]
    policy_matrix = np.zeros((len(all_states), len(penalties)))

    for j, pen in enumerate(penalties):
        pol = _run_value_iteration(reward_arrive=100, reward_divert=pen, reward_bad=-80)
        for i, (wp, w) in enumerate(all_states):
            policy_matrix[i, j] = 1 if pol[(wp, w)] == 'continue' else 0

    ytick_labels = [f"{WAYPOINT_NAMES[wp]}, {WEATHER_CONDITIONS[w]}" for (wp, w) in all_states]
    cmap = mcolors.ListedColormap(['#e74c3c', '#2ecc71'])
    ax.imshow(policy_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1,
              extent=[penalties[0], penalties[-1], len(all_states) - 0.5, -0.5])
    ax.set_yticks(range(len(all_states)))
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_xlabel("Divert penalty  R(divert)", fontsize=10)
    ax.set_title("Policy  (green=continue, red=divert)")
    ax.axvline(-20, color='white', linestyle='--', linewidth=1.5, label='baseline (−20)')
    ax.legend(fontsize=8)

    # Right: flip points for selected states
    ax2 = axes[1]
    colors_sel = ['#8e44ad', '#2980b9', '#e67e22', '#27ae60', '#c0392b']
    for (wp, w, label), color in zip(states_of_interest, colors_sel):
        actions = []
        for pen in penalties:
            pol = _run_value_iteration(reward_arrive=100, reward_divert=pen, reward_bad=-80)
            actions.append(1 if pol[(wp, w)] == 'continue' else 0)
        ax2.plot(penalties, actions, '-', color=color, linewidth=2.5, label=label)

    ax2.set_xlabel("Divert penalty  R(divert)", fontsize=10)
    ax2.set_ylabel("Action  (1=continue, 0=divert)", fontsize=10)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['divert', 'continue'])
    ax2.legend(fontsize=8, loc='center right')
    ax2.set_title("Policy flip lines — selected states")
    ax2.axvline(-20, color='gray', linestyle='--', linewidth=1, label='baseline')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/sensitivity_divert.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/sensitivity_divert.png")


def sensitivity_bad_weather_cost():
    """
    Sweep the bad-weather continue cost from −20 to −150.
    """
    from scenario import WAYPOINT_NAMES, N_WAYPOINTS, WEATHER_CONDITIONS

    costs = np.linspace(-20, -150, 80)
    all_states = [(wp, w) for wp in range(N_WAYPOINTS) for w in range(3)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sensitivity Analysis — Bad-Weather Continue Cost", fontsize=13, fontweight='bold')

    # Heatmap
    ax = axes[0]
    policy_matrix = np.zeros((len(all_states), len(costs)))
    for j, cost in enumerate(costs):
        pol = _run_value_iteration(reward_arrive=100, reward_divert=-20, reward_bad=cost)
        for i, (wp, w) in enumerate(all_states):
            policy_matrix[i, j] = 1 if pol[(wp, w)] == 'continue' else 0

    ytick_labels = [f"{WAYPOINT_NAMES[wp]}, {WEATHER_CONDITIONS[w]}" for (wp, w) in all_states]
    cmap = mcolors.ListedColormap(['#e74c3c', '#2ecc71'])
    ax.imshow(policy_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1,
              extent=[costs[0], costs[-1], len(all_states) - 0.5, -0.5])
    ax.set_yticks(range(len(all_states)))
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_xlabel("Bad-weather cost  R(continue | bad)", fontsize=10)
    ax.set_title("Policy  (green=continue, red=divert)")
    ax.axvline(-80, color='white', linestyle='--', linewidth=1.5, label='baseline (−80)')
    ax.legend(fontsize=8)

    # Flip lines for bad-weather states
    ax2 = axes[1]
    bad_states = [(wp, 2, f"{WAYPOINT_NAMES[wp]}, bad") for wp in range(N_WAYPOINTS)]
    marg_states = [(2, 1, 'KTRM, marginal'), (1, 1, 'KBXK, marginal')]
    colors_b = ['#8e44ad', '#2980b9', '#e67e22', '#27ae60']
    colors_m = ['#c0392b', '#16a085']

    for (wp, w, label), color in zip(bad_states, colors_b):
        actions = [1 if _run_value_iteration(100, -20, c)[(wp, w)] == 'continue' else 0
                   for c in costs]
        ax2.plot(costs, actions, '-', color=color, linewidth=2.5, label=label)

    for (wp, w, label), color in zip(marg_states, colors_m):
        actions = [1 if _run_value_iteration(100, -20, c)[(wp, w)] == 'continue' else 0
                   for c in costs]
        ax2.plot(costs, actions, '--', color=color, linewidth=2, label=label)

    ax2.set_xlabel("Bad-weather cost  R(continue | bad)", fontsize=10)
    ax2.set_ylabel("Action  (1=continue, 0=divert)", fontsize=10)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['divert', 'continue'])
    ax2.legend(fontsize=8)
    ax2.set_title("Policy flip lines")
    ax2.axvline(-80, color='gray', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/sensitivity_bad_cost.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/sensitivity_bad_cost.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Belief threshold visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_belief_thresholds():
    """
    For each waypoint, shade the simplex by optimal POMDP action.
    Green = continue, red = divert.
    The boundary between them is the policy threshold.
    """
    from pomdp_solver import solve_pomdp, get_action

    print("Solving POMDP for threshold plot...")
    gamma_per_wp = solve_pomdp()

    from scenario import WAYPOINT_NAMES, N_WAYPOINTS

    # Sample a dense grid of belief points on the simplex
    N = 80
    fig, axes = plt.subplots(1, N_WAYPOINTS, figsize=(16, 4))
    fig.suptitle("POMDP Policy Threshold — Belief Simplex per Waypoint\n"
                 "Green = continue, Red = divert. Boundary = decision threshold.",
                 fontsize=12, fontweight='bold')

    verts_2d = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])

    for wp, ax in enumerate(axes):
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(WAYPOINT_NAMES[wp], fontsize=11, fontweight='bold')

        # Draw triangle outline
        tri = plt.Polygon(verts_2d, fill=False, edgecolor='#333', linewidth=1.5, zorder=3)
        ax.add_patch(tri)

        # Labels
        off = 0.06
        ax.text(-off, -off, 'Good', ha='center', fontsize=8, color=WEATHER_COLORS[0], fontweight='bold')
        ax.text(1 + off, -off, 'Marginal', ha='center', fontsize=8, color=WEATHER_COLORS[1], fontweight='bold')
        ax.text(0.5, np.sqrt(3)/2 + off, 'Bad', ha='center', fontsize=8, color=WEATHER_COLORS[2], fontweight='bold')

        # Sample grid: points on the simplex
        points_2d = []
        actions = []
        for i in range(N + 1):
            for j in range(N + 1 - i):
                k = N - i - j
                b = np.array([i, j, k], dtype=float) / N
                if b.sum() < 0.99:
                    continue
                act, _, _, _ = get_action(b, gamma_per_wp[wp])
                px, py = _to_2d(b)
                points_2d.append((px, py))
                actions.append(act)

        points_2d = np.array(points_2d)
        actions = np.array(actions)

        # Scatter with tiny markers
        cont_mask = actions == 0
        div_mask  = actions == 1
        ax.scatter(points_2d[cont_mask, 0], points_2d[cont_mask, 1],
                   c='#2ecc71', s=8, alpha=0.7, zorder=2)
        ax.scatter(points_2d[div_mask, 0],  points_2d[div_mask, 1],
                   c='#e74c3c', s=8, alpha=0.7, zorder=2)

        # Mark baseline belief (uniform prior)
        px, py = _to_2d([1/3, 1/3, 1/3])
        ax.plot(px, py, 'k*', markersize=10, zorder=4, label='uniform prior')

        # Mark the three pure states
        for b_pure, label in zip([[1,0,0],[0,1,0],[0,0,1]], WEATHER_LABELS):
            px2, py2 = _to_2d(b_pure)
            act_pure, _, _, _ = get_action(np.array(b_pure, float), gamma_per_wp[wp])
            ax.plot(px2, py2, 'o', color='#2ecc71' if act_pure == 0 else '#e74c3c',
                    markersize=8, zorder=4, markeredgecolor='k', markeredgewidth=0.8)

        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.12, 1.0)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=9, label='continue'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=9, label='divert'),
        Line2D([0], [0], marker='*', color='k', markersize=10, label='uniform prior'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("plots/belief_thresholds.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plots/belief_thresholds.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('plots', exist_ok=True)

    print("=== Sensitivity: divert penalty ===")
    sensitivity_divert_penalty()

    print("\n=== Sensitivity: bad-weather cost ===")
    sensitivity_bad_weather_cost()

    print("\n=== Belief threshold simplex ===")
    plot_belief_thresholds()

    print("\nDone. Opening plots...")
