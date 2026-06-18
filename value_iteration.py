"""
Value iteration for solving the aviation weather MDP.

Algorithm (from Kochenderfer et al., Ch. 7):

    Initialize U(s) = 0 for all s
    Repeat until convergence:
        For each state s:
            U(s) <- max_a [ R(s,a) + γ * Σ_{s'} T(s'|s,a) * U(s') ]

The inner term — R(s,a) + γ * Σ T(s'|s,a) * U(s') — is the Q-value Q(s,a).
Taking the max over actions gives the updated value for state s (Bellman backup).

Convergence is guaranteed because the Bellman backup is a contraction mapping.
We stop when the Bellman residual (max change across all states) falls below
a tolerance threshold.
"""

import numpy as np
from scenario import ACTIONS, N_ACTIONS


def q_value(states, transition_fn, reward_fn, U, s, a, gamma):
    """
    Q(s, a) = R(s, a) + γ * Σ_{s'} T(s'|s,a) * U(s')

    The expected return from taking action a in state s,
    then following the optimal policy thereafter.
    """
    q = reward_fn(s, a)
    for i, s_prime in enumerate(states):
        p = transition_fn(states, s, a, s_prime)
        if p > 0:
            q += gamma * p * U[i]
    return q


def bellman_backup(states, transition_fn, reward_fn, U, s, gamma):
    """
    Apply the Bellman backup to state s.
    Returns (best_value, best_action_index).
    """
    best_value = -np.inf
    best_action = None

    for a in range(N_ACTIONS):
        q = q_value(states, transition_fn, reward_fn, U, s, a, gamma)
        if q > best_value:
            best_value = q
            best_action = a

    return best_value, best_action


def value_iteration(states, transition_fn, reward_fn, gamma,
                    max_iterations=500, tolerance=1e-6):
    """
    Run value iteration to find U* and π*.

    Parameters
    ----------
    states        : list of all states
    transition_fn : T(states, s, a, s') -> float
    reward_fn     : R(s, a) -> float
    gamma         : discount factor
    max_iterations: maximum number of Bellman backup sweeps
    tolerance     : convergence threshold on Bellman residual

    Returns
    -------
    U         : optimal value function, shape (len(states),)
    policy    : dict mapping state -> optimal action index
    residuals : list of Bellman residuals per iteration
    """
    n = len(states)
    U = np.zeros(n)
    residuals = []

    for iteration in range(max_iterations):
        U_new = np.zeros(n)

        for i, s in enumerate(states):
            U_new[i], _ = bellman_backup(states, transition_fn, reward_fn, U, s, gamma)

        residual = float(np.max(np.abs(U_new - U)))
        residuals.append(residual)
        U = U_new

        if residual < tolerance:
            print(f"Converged in {iteration + 1} iterations (residual={residual:.2e})")
            break
    else:
        print(f"Reached max iterations ({max_iterations}), residual={residuals[-1]:.2e}")

    # Extract optimal policy from converged U
    policy = {}
    for s in states:
        _, best_action = bellman_backup(states, transition_fn, reward_fn, U, s, gamma)
        policy[s] = best_action

    return U, policy, residuals
