"""
MDP formulation for the aviation weather go/no-go problem.

State space:
  - In-flight: (waypoint_index, weather_index)  for all waypoints and weather conditions
  - Terminal:  'arrived'  — pilot successfully reached the destination
               'diverted' — pilot chose to divert at some point

Action space:
  - CONTINUE (0): fly to the next waypoint
  - DIVERT   (1): land at the nearest alternate airport

Transition function T(s, a, s'):
  - Diverting always leads to 'diverted'
  - Continuing from the last waypoint leads to 'arrived'
  - Continuing from any other waypoint advances to (wp+1, new_weather),
    where new_weather is drawn from WEATHER_TRANSITION[current_weather]
  - Terminal states are absorbing (stay forever once reached)

Reward function R(s, a):
  - Arriving at the destination:     +100  (mission complete)
  - Diverting:                        -20  (safe but costly)
  - Continuing in good weather:        +2  (smooth progress)
  - Continuing in marginal weather:    -5  (uncomfortable and risky)
  - Continuing in bad weather:        -80  (dangerous)
"""

import numpy as np
from scenario import (
    N_WAYPOINTS, N_WEATHER, WEATHER_CONDITIONS,
    CONTINUE, DIVERT,
    WEATHER_TRANSITION, WEATHER_CONTINUE_REWARDS,
    REWARD_ARRIVE, REWARD_DIVERT,
)


def build_states():
    """
    Build and return the ordered list of all states.

    In-flight states come first (waypoint 0 to N-1, weather 0 to N_WEATHER-1),
    followed by the two terminal states.
    """
    states = []
    for wp in range(N_WAYPOINTS):
        for w in range(N_WEATHER):
            states.append((wp, w))
    states.append('arrived')
    states.append('diverted')
    return states


def transition(states, s, a, s_prime):
    """
    T(s' | s, a): probability of reaching state s_prime from s after action a.
    """
    # Terminal states are absorbing
    if s in ('arrived', 'diverted'):
        return 1.0 if s_prime == s else 0.0

    wp, weather = s

    if a == DIVERT:
        return 1.0 if s_prime == 'diverted' else 0.0

    # Action: CONTINUE
    if wp == N_WAYPOINTS - 1:
        # Last waypoint — continuing means arriving
        return 1.0 if s_prime == 'arrived' else 0.0

    # Mid-route — advance to next waypoint with stochastic weather
    if s_prime in ('arrived', 'diverted'):
        return 0.0

    next_wp, next_weather = s_prime
    if next_wp != wp + 1:
        return 0.0

    return float(WEATHER_TRANSITION[weather, next_weather])


def reward(s, a):
    """
    R(s, a): expected reward for taking action a in state s.
    """
    if s in ('arrived', 'diverted'):
        return 0.0

    wp, weather = s

    if a == DIVERT:
        return REWARD_DIVERT

    # Action: CONTINUE
    step_reward = WEATHER_CONTINUE_REWARDS[weather]

    if wp == N_WAYPOINTS - 1:
        # Arrival bonus on top of the weather penalty/reward
        return REWARD_ARRIVE + step_reward

    return step_reward
