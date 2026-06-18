"""
Visualizations for the aviation weather MDP.

Four plots:
  1. value_function  — heatmap of U*(s) over (waypoint × weather)
  2. policy_map      — optimal action at each state (continue vs. divert)
  3. convergence     — Bellman residual per iteration (log scale)
  4. simulation      — one sample flight path showing decisions made
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from scenario import (
    WAYPOINT_NAMES, WEATHER_CONDITIONS, N_WAYPOINTS, N_WEATHER,
    CONTINUE, DIVERT, ACTIONS,
)


# ---------------------------------------------------------------------------
# 1. Value function heatmap
# ---------------------------------------------------------------------------

def plot_value_function(states, U, save_path=None):
    """Heatmap of U*(s) over all (waypoint, weather) states."""
    grid = np.full((N_WEATHER, N_WAYPOINTS), np.nan)

    for i, s in enumerate(states):
        if isinstance(s, tuple):
            wp, weather = s
            grid[weather, wp] = U[i]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, cmap='RdYlGn', aspect='auto',
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))

    ax.set_xticks(range(N_WAYPOINTS))
    ax.set_xticklabels(WAYPOINT_NAMES, fontsize=12)
    ax.set_yticks(range(N_WEATHER))
    ax.set_yticklabels(WEATHER_CONDITIONS, fontsize=12)
    ax.set_xlabel('Waypoint', fontsize=12)
    ax.set_ylabel('Weather Condition', fontsize=12)
    ax.set_title('Optimal Value Function  U*(s)', fontsize=14, fontweight='bold')

    for wp in range(N_WAYPOINTS):
        for w in range(N_WEATHER):
            v = grid[w, wp]
            if not np.isnan(v):
                ax.text(wp, w, f'{v:.1f}', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='black' if abs(v) < 60 else 'white')

    plt.colorbar(im, ax=ax, label='Value U*(s)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Policy map
# ---------------------------------------------------------------------------

def plot_policy(states, policy, save_path=None):
    """Color-coded grid showing the optimal action at each state."""
    grid = np.full((N_WEATHER, N_WAYPOINTS), np.nan)

    for s, a in policy.items():
        if isinstance(s, tuple):
            wp, weather = s
            grid[weather, wp] = float(a)

    cmap = mcolors.ListedColormap(['#4CAF50', '#E53935'])  # green=continue, red=divert
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(N_WAYPOINTS))
    ax.set_xticklabels(WAYPOINT_NAMES, fontsize=12)
    ax.set_yticks(range(N_WEATHER))
    ax.set_yticklabels(WEATHER_CONDITIONS, fontsize=12)
    ax.set_xlabel('Waypoint', fontsize=12)
    ax.set_ylabel('Weather Condition', fontsize=12)
    ax.set_title('Optimal Policy  π*(s)', fontsize=14, fontweight='bold')

    for wp in range(N_WAYPOINTS):
        for w in range(N_WEATHER):
            a = grid[w, wp]
            if not np.isnan(a):
                label = ACTIONS[int(a)]
                ax.text(wp, w, label, ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white')

    legend = [
        mpatches.Patch(color='#4CAF50', label='Continue'),
        mpatches.Patch(color='#E53935', label='Divert'),
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(residuals, save_path=None):
    """Log-scale plot of Bellman residual over iterations."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(residuals, color='steelblue', linewidth=2, label='Bellman residual')
    ax.axhline(y=1e-6, color='#E53935', linestyle='--', linewidth=1.5, label='Tolerance (1e-6)')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Bellman Residual (log scale)', fontsize=12)
    ax.set_title('Value Iteration Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Simulate and plot a sample flight
# ---------------------------------------------------------------------------

def simulate_flight(states, policy, initial_weather=0, seed=42):
    """
    Simulate a single flight under the optimal policy.

    Returns a list of dicts: {waypoint, weather, action}
    """
    from scenario import WEATHER_TRANSITION

    rng = np.random.default_rng(seed)
    trajectory = []
    wp = 0
    weather = initial_weather

    while True:
        s = (wp, weather)
        a = policy.get(s, DIVERT)
        trajectory.append({
            'waypoint': WAYPOINT_NAMES[wp],
            'weather':  WEATHER_CONDITIONS[weather],
            'action':   ACTIONS[a],
        })

        if a == DIVERT:
            trajectory.append({'waypoint': 'ALTERNATE', 'weather': '—', 'action': 'diverted'})
            break

        if wp == N_WAYPOINTS - 1:
            trajectory.append({'waypoint': 'ARRIVED', 'weather': WEATHER_CONDITIONS[weather], 'action': 'success'})
            break

        weather = int(rng.choice(N_WEATHER, p=WEATHER_TRANSITION[weather]))
        wp += 1

    return trajectory


def plot_simulation(trajectory, title='Simulated Flight', save_path=None):
    """Timeline diagram of waypoints, weather, and decisions."""
    WEATHER_COLORS = {'good': '#4CAF50', 'marginal': '#FF9800', 'bad': '#E53935', '—': '#9E9E9E'}
    ACTION_COLORS  = {'continue': '#1565C0', 'divert': '#E53935', 'diverted': '#7B1FA2', 'success': '#2E7D32'}

    n = len(trajectory)
    fig, ax = plt.subplots(figsize=(max(10, n * 2.2), 3.5))

    for i, step in enumerate(trajectory):
        wp     = step['waypoint']
        weather = step['weather']
        action  = step['action']

        bg = WEATHER_COLORS.get(weather, '#9E9E9E')
        rect = mpatches.FancyBboxPatch((i - 0.38, -0.35), 0.76, 0.70,
                                        boxstyle='round,pad=0.05',
                                        facecolor=bg, edgecolor='white',
                                        linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)

        ax.text(i, 0.05, wp,     ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(i, -0.22, weather, ha='center', va='center', fontsize=8,  color='white')
        ax.text(i, 0.58, action,  ha='center', va='center', fontsize=9,
                color=ACTION_COLORS.get(action, 'black'), fontweight='bold')

        if i < n - 1:
            ax.annotate('', xy=(i + 0.42, 0), xytext=(i + 0.38, 0),
                        arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.6, 0.9)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
