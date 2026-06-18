"""
Aviation Weather POMDP — main entry point.

Solves the POMDP extension, prints the optimal policy at representative
belief states, simulates three flights, and exports all data to
pomdp_data.json for the interactive widget.
"""

import json
import os
import numpy as np

from scenario import (
    WAYPOINT_NAMES, WEATHER_CONDITIONS, ACTIONS,
    DISCOUNT_FACTOR, WEATHER_TRANSITION,
)
from pomdp import (
    OBS_MODEL, OBS_LABELS, OBS_CATEGORY, P_HIGH_CONF,
    N_OBS, predict_belief, belief_update,
)
from pomdp_solver import solve_pomdp, get_action, simulate_pomdp

os.makedirs('plots', exist_ok=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print("=" * 65)
print("  Aviation Weather Go/No-Go — POMDP with Multiple Sensors")
print("=" * 65)
print()
print("Observation model:")
print()
print("  METAR category sensor  P(reported | true):")
print(f"  {'':14}  {'→ good':>8}  {'→ marginal':>10}  {'→ bad':>7}")
for i, w in enumerate(WEATHER_CONDITIONS):
    print(
        f"  true {w:<10}  "
        f"{OBS_CATEGORY[i, 0]:>8.2f}  "
        f"{OBS_CATEGORY[i, 1]:>10.2f}  "
        f"{OBS_CATEGORY[i, 2]:>7.2f}"
    )
print()
print("  Confidence sensor  P(high-conf | true):")
for i, w in enumerate(WEATHER_CONDITIONS):
    print(f"  true {w:<10}  {P_HIGH_CONF[i]:.2f}")
print()

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

print("Solving POMDP by backward induction...")
gamma_per_wp = solve_pomdp()
print()

# ---------------------------------------------------------------------------
# Print policy at key beliefs
# ---------------------------------------------------------------------------

key_beliefs = [
    (np.array([1.0, 0.0, 0.0]), "certain good"),
    (np.array([0.0, 1.0, 0.0]), "certain marginal"),
    (np.array([0.0, 0.0, 1.0]), "certain bad"),
    (np.array([1/3, 1/3, 1/3]), "uniform"),
    (np.array([0.6, 0.3, 0.1]), "likely good"),
    (np.array([0.1, 0.3, 0.6]), "likely bad"),
]

print(f"{'Waypoint':<8}  {'Belief (G/M/B)':<24}  {'Scenario':<18}  "
      f"{'Action':<10}  {'V(b)':>7}  {'Q(cont)':>8}  {'Q(div)':>7}")
print("-" * 92)

for wp, wp_name in enumerate(WAYPOINT_NAMES):
    for b, label in key_beliefs:
        a, v, qc, qd = get_action(b, gamma_per_wp[wp])
        bstr = f"({b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f})"
        print(
            f"{wp_name:<8}  {bstr:<24}  {label:<18}  "
            f"{ACTIONS[a]:<10}  {v:>7.2f}  {qc:>8.2f}  {qd:>7.2f}"
        )
    print()

# ---------------------------------------------------------------------------
# Simulate flights
# ---------------------------------------------------------------------------

print("Simulated flights under POMDP policy:")
for w_idx, w_name in enumerate(WEATHER_CONDITIONS):
    print(f"\n  True starting weather: {w_name}")
    traj = simulate_pomdp(gamma_per_wp, initial_weather=w_idx, seed=42)
    for step in traj:
        if step['belief'] is not None:
            b = step['belief']
            bstr = f"b=({b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f})"
        else:
            bstr = ""
        print(
            f"    {step['waypoint']:<12}  "
            f"true={step['true_weather']:<10}  "
            f"obs={step['obs']:<25}  "
            f"{bstr:<26}  → {step['action']}"
        )

# ---------------------------------------------------------------------------
# Export data for interactive widget
# ---------------------------------------------------------------------------

export = {
    'waypoints':    WAYPOINT_NAMES,
    'weather':      WEATHER_CONDITIONS,
    'actions':      ACTIONS,
    'obs_labels':   OBS_LABELS,
    'obs_model':    OBS_MODEL.tolist(),
    'obs_category': OBS_CATEGORY.tolist(),
    'p_high_conf':  P_HIGH_CONF.tolist(),
    'transition':   WEATHER_TRANSITION.tolist(),
    'discount':     DISCOUNT_FACTOR,
    'rewards': {
        'arrive': 100.0,
        'divert': -20.0,
        'continue': [2.0, -5.0, -80.0],
    },
    'gamma_per_wp': [
        [alpha.tolist() for alpha in gw]
        for gw in gamma_per_wp
    ],
}

out_path = 'pomdp_data.json'
with open(out_path, 'w') as f:
    json.dump(export, f, indent=2)

print(f"\nExported POMDP data → {out_path}")
