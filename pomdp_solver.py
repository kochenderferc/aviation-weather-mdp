"""
POMDP solver: backward induction with alpha vectors.

Because the pilot always knows which waypoint they are at, the problem
decomposes into a sequence of independent POMDP stages — one per waypoint.
At each stage the value function is a convex piecewise-linear function
over the 2-simplex of weather beliefs, represented as a set of alpha vectors:

    V_wp(b) = max_{α ∈ Γ_wp}  b · α

We solve by working backwards from the last waypoint (KSAN) to the first
(KPHX), computing Γ_wp from Γ_{wp+1} using a PBVI-style backup.

Bellman backup for action 'continue' at waypoint wp:
    For each o ∈ observations, α' ∈ Γ_{wp+1}:
        φ_{o,α'}(w) = Σ_{w'} T(w'|w) · O(o|w') · α'(w')

    For a given belief b, pick the best α' per observation:
        combo*(b) = argmax_{α'} b · φ_{o,α'}   for each o

    The resulting alpha vector:
        α_combo(w) = R_continue(w) + γ · Σ_o φ_{o, combo*(b)[o]}(w)

    PBVI: sample ~400 belief points, collect unique combos → prune dominated.
"""

import numpy as np
from scenario import (
    N_WAYPOINTS, WAYPOINT_NAMES,
    WEATHER_CONTINUE_REWARDS, REWARD_ARRIVE, REWARD_DIVERT,
    DISCOUNT_FACTOR, N_WEATHER, WEATHER_TRANSITION,
)
from pomdp import OBS_MODEL, N_OBS


# ---------------------------------------------------------------------------
# Helper: phi vector
# ---------------------------------------------------------------------------

def _phi(alpha_prime, obs_idx):
    """
    φ_{o,α'}(w) = Σ_{w'} T(w'|w) · O(o|w') · α'(w')
               = WEATHER_TRANSITION @ (OBS_MODEL[:, o] * α')

    Shape: (N_WEATHER,)
    """
    return WEATHER_TRANSITION @ (OBS_MODEL[:, obs_idx] * alpha_prime)


# ---------------------------------------------------------------------------
# Belief point sampling
# ---------------------------------------------------------------------------

def _sample_beliefs(n=400, seed=0):
    """Uniform samples from the (N_WEATHER-1)-simplex via Dirichlet."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(N_WEATHER), size=n)


# ---------------------------------------------------------------------------
# Alpha vector backup
# ---------------------------------------------------------------------------

def _backup_continue(gamma_next, wp, gamma):
    """
    Compute alpha vectors for action 'continue' at waypoint wp.

    Parameters
    ----------
    gamma_next : list of np.arrays — alpha vectors from the next waypoint
                 (None if wp is the last waypoint)
    wp         : waypoint index
    gamma      : discount factor

    Returns
    -------
    list of np.arrays, each shape (N_WEATHER,)
    """
    r = np.array(WEATHER_CONTINUE_REWARDS, dtype=float)

    if wp == N_WAYPOINTS - 1:
        # Continuing from the last waypoint = arriving. No future value.
        return [r + REWARD_ARRIVE]

    n_alpha = len(gamma_next)

    # Precompute phi[o][k] = φ_{o, gamma_next[k]}   shape (N_WEATHER,)
    phi = [
        [_phi(gamma_next[k], o) for k in range(n_alpha)]
        for o in range(N_OBS)
    ]

    beliefs = _sample_beliefs()
    alpha_set = []
    seen = set()

    for b in beliefs:
        # For each observation, find which alpha' in Γ_{wp+1} is best
        combo = tuple(
            int(np.argmax([float(b @ phi[o][k]) for k in range(n_alpha)]))
            for o in range(N_OBS)
        )
        if combo in seen:
            continue
        seen.add(combo)

        # Build the alpha vector for this combination
        alpha = r.copy()
        for o in range(N_OBS):
            alpha = alpha + gamma * phi[o][combo[o]]
        alpha_set.append(alpha)

    return alpha_set if alpha_set else [r]


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_pomdp(gamma=DISCOUNT_FACTOR):
    """
    Solve the POMDP by backward induction over waypoints.

    Returns
    -------
    gamma_per_wp : list of length N_WAYPOINTS
                   gamma_per_wp[wp] = list of alpha vectors (np.arrays of
                   shape (N_WEATHER,)) for waypoint wp.
                   gamma_per_wp[wp][0] is always the 'divert' alpha vector.
    """
    gamma_per_wp = [None] * N_WAYPOINTS
    gamma_next = None

    for wp in range(N_WAYPOINTS - 1, -1, -1):
        alpha_div = np.full(N_WEATHER, REWARD_DIVERT, dtype=float)
        alpha_cont_set = _backup_continue(gamma_next, wp, gamma)

        gamma_per_wp[wp] = [alpha_div] + alpha_cont_set
        gamma_next = gamma_per_wp[wp]

        n = len(gamma_per_wp[wp])
        print(f"  {WAYPOINT_NAMES[wp]}: {n} alpha vector{'s' if n != 1 else ''}")

    return gamma_per_wp


# ---------------------------------------------------------------------------
# Policy query
# ---------------------------------------------------------------------------

def get_action(b, gamma_wp):
    """
    Optimal action and values at belief b.

    Parameters
    ----------
    b        : belief vector, shape (N_WEATHER,)
    gamma_wp : list of alpha vectors for this waypoint

    Returns
    -------
    action    : 0 = continue, 1 = divert
    value     : V*(b)
    q_continue: Q(b, continue)
    q_divert  : Q(b, divert)
    """
    q_div = float(b @ gamma_wp[0])
    if len(gamma_wp) > 1:
        q_cont = float(max(b @ alpha for alpha in gamma_wp[1:]))
    else:
        q_cont = -np.inf

    if q_cont >= q_div:
        return 0, q_cont, q_cont, q_div
    else:
        return 1, q_div, q_cont, q_div


# ---------------------------------------------------------------------------
# Flight simulation under POMDP policy
# ---------------------------------------------------------------------------

def simulate_pomdp(gamma_per_wp, initial_weather=None, seed=42):
    """
    Simulate one flight under the optimal POMDP policy.

    The pilot starts with a uniform prior over weather. At each waypoint
    they receive a noisy observation, update their belief, then act.

    Parameters
    ----------
    gamma_per_wp   : output of solve_pomdp()
    initial_weather: index of true starting weather (None = random)
    seed           : RNG seed

    Returns
    -------
    trajectory : list of dicts, one per waypoint visited
    """
    from scenario import WEATHER_CONDITIONS, ACTIONS, WEATHER_TRANSITION
    from pomdp import OBS_LABELS, belief_update, predict_belief

    rng = np.random.default_rng(seed)

    if initial_weather is None:
        true_w = int(rng.integers(N_WEATHER))
    else:
        true_w = initial_weather

    # Pilot starts with a uniform prior (no information yet)
    b = np.ones(N_WEATHER) / N_WEATHER

    trajectory = []

    for wp in range(N_WAYPOINTS):
        # --- Observe at this waypoint ---
        obs_idx = int(rng.choice(N_OBS, p=OBS_MODEL[true_w]))

        # Update belief with this observation
        b_new = OBS_MODEL[:, obs_idx] * b
        norm = b_new.sum()
        b = b_new / norm if norm > 1e-12 else b

        # --- Act ---
        action, val, q_cont, q_div = get_action(b, gamma_per_wp[wp])

        trajectory.append({
            'waypoint':      WAYPOINT_NAMES[wp],
            'true_weather':  WEATHER_CONDITIONS[true_w],
            'belief':        b.copy(),
            'obs':           OBS_LABELS[obs_idx],
            'action':        ACTIONS[action],
            'value':         val,
            'q_continue':    q_cont,
            'q_divert':      q_div,
        })

        if action == 1:   # divert
            trajectory.append({
                'waypoint': 'ALTERNATE', 'true_weather': '—',
                'belief': None, 'obs': '—', 'action': 'diverted',
                'value': REWARD_DIVERT, 'q_continue': None, 'q_divert': None,
            })
            break

        if wp == N_WAYPOINTS - 1:   # arrived
            trajectory.append({
                'waypoint': 'ARRIVED', 'true_weather': WEATHER_CONDITIONS[true_w],
                'belief': None, 'obs': '—', 'action': 'success',
                'value': REWARD_ARRIVE, 'q_continue': None, 'q_divert': None,
            })
            break

        # --- Transition: sample next true weather, predict belief ---
        true_w = int(rng.choice(N_WEATHER, p=WEATHER_TRANSITION[true_w]))
        b = predict_belief(b)   # propagate belief through transition model

    return trajectory
