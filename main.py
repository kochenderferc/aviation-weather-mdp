"""
Aviation Weather Go/No-Go MDP — main entry point.

Builds the MDP, solves it with value iteration, prints the optimal policy,
and saves all visualizations to the plots/ directory.
"""

import os
from scenario import WAYPOINT_NAMES, WEATHER_CONDITIONS, ACTIONS, DISCOUNT_FACTOR
from mdp import build_states, transition, reward
from value_iteration import value_iteration
from visualize import (
    plot_value_function,
    plot_policy,
    plot_convergence,
    simulate_flight,
    plot_simulation,
)

os.makedirs('plots', exist_ok=True)

# ---------------------------------------------------------------------------
# Build state space
# ---------------------------------------------------------------------------

states = build_states()
n_inflight = sum(1 for s in states if isinstance(s, tuple))

print("=" * 55)
print("  Aviation Weather Go/No-Go MDP")
print("=" * 55)
print(f"Route:    {' → '.join(WAYPOINT_NAMES)}")
print(f"States:   {len(states)} total ({n_inflight} in-flight + 2 terminal)")
print(f"Actions:  {ACTIONS}")
print(f"Gamma:    {DISCOUNT_FACTOR}")
print()

# ---------------------------------------------------------------------------
# Solve with value iteration
# ---------------------------------------------------------------------------

print("Running value iteration...")
U, policy, residuals = value_iteration(
    states=states,
    transition_fn=transition,
    reward_fn=reward,
    gamma=DISCOUNT_FACTOR,
    max_iterations=500,
    tolerance=1e-6,
)
print()

# ---------------------------------------------------------------------------
# Print optimal policy
# ---------------------------------------------------------------------------

print(f"{'State':<28} {'Action':<10} {'Value':>8}")
print("-" * 50)
for i, s in enumerate(states):
    if isinstance(s, tuple):
        wp_idx, w_idx = s
        label = f"({WAYPOINT_NAMES[wp_idx]}, {WEATHER_CONDITIONS[w_idx]})"
        action = ACTIONS[policy[s]]
        print(f"{label:<28} {action:<10} {U[i]:>8.2f}")
print()

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

print("Generating plots...")
plot_value_function(states, U,   save_path='plots/value_function.png')
plot_policy(states, policy,      save_path='plots/policy_map.png')
plot_convergence(residuals,      save_path='plots/convergence.png')

# Simulate flights from each starting weather condition
print("\nSimulating flights under optimal policy:")
for w_idx, w_name in enumerate(WEATHER_CONDITIONS):
    traj = simulate_flight(states, policy, initial_weather=w_idx, seed=42)
    print(f"\n  Starting weather: {w_name}")
    for step in traj:
        print(f"    {step['waypoint']:<12}  weather={step['weather']:<10}  → {step['action']}")
    plot_simulation(
        traj,
        title=f'Simulated Flight — Starting in {w_name.capitalize()} Weather',
        save_path=f'plots/simulation_{w_name}.png',
    )

print("\nAll plots saved to plots/")
