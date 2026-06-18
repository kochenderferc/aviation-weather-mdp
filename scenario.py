"""
Aviation Weather Go/No-Go Decision Problem — Scenario Definition

A pilot is flying from Phoenix (KPHX) to San Diego (KSAN) via two intermediate
waypoints. At each waypoint, the pilot observes weather ahead and must decide:
continue to the next waypoint, or divert to the nearest alternate airport.

This module defines all scenario-specific parameters:
  - Route and waypoints
  - Weather conditions and transition probabilities
  - Reward structure
  - Discount factor
"""

import numpy as np

# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

WAYPOINT_NAMES = ['KPHX', 'KBXK', 'KTRM', 'KSAN']  # Phoenix -> Blythe -> Thermal -> San Diego
N_WAYPOINTS = len(WAYPOINT_NAMES)

# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

WEATHER_CONDITIONS = ['good', 'marginal', 'bad']
N_WEATHER = len(WEATHER_CONDITIONS)
GOOD, MARGINAL, BAD = 0, 1, 2

# Weather transition matrix: T_weather[current, next]
# Represents how weather evolves between waypoints when the pilot continues.
# Weather tends to persist but can improve or worsen.
WEATHER_TRANSITION = np.array([
    [0.70, 0.25, 0.05],   # good     -> [good, marginal, bad]
    [0.30, 0.40, 0.30],   # marginal -> [good, marginal, bad]
    [0.05, 0.25, 0.70],   # bad      -> [good, marginal, bad]
])

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

ACTIONS = ['continue', 'divert']
N_ACTIONS = len(ACTIONS)
CONTINUE, DIVERT = 0, 1

# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

REWARD_ARRIVE          =  100.0   # successfully completing the flight
REWARD_DIVERT          =  -20.0   # diverting: safe but inconvenient (time, cost)
REWARD_CONTINUE_GOOD   =    2.0   # making progress in good conditions
REWARD_CONTINUE_MARGINAL = -5.0   # flying in marginal weather: uncomfortable and risky
REWARD_CONTINUE_BAD    =  -80.0   # flying in bad weather: dangerous

WEATHER_CONTINUE_REWARDS = [
    REWARD_CONTINUE_GOOD,
    REWARD_CONTINUE_MARGINAL,
    REWARD_CONTINUE_BAD,
]

# ---------------------------------------------------------------------------
# Discount factor
# ---------------------------------------------------------------------------

# γ = 0.95: future rewards are worth 95% of immediate rewards.
# Reflects mild time-preference and helps the algorithm converge.
DISCOUNT_FACTOR = 0.95
