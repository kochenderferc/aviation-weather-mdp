"""
POMDP extension: observation model and belief update.

The pilot can no longer observe true weather directly. At each waypoint
they receive two noisy sensor readings:

  1. METAR category  — a categorical reading (good / marginal / bad) that
                       can be wrong. E.g. true bad weather is reported as
                       marginal ~25 % of the time.

  2. Confidence score — binary (high / low), indicating how much the
                        sensors agree with one another. Marginal weather
                        produces low confidence more often because the
                        signal is ambiguous.

Observation space: 6 combinations of (category × confidence).
Indexing: obs = 2 * category + confidence  (conf: 0=low, 1=high)

    0  good / low-conf      1  good / high-conf
    2  marginal / low-conf  3  marginal / high-conf
    4  bad / low-conf       5  bad / high-conf
"""

import numpy as np
from scenario import WEATHER_TRANSITION, N_WEATHER

# ---------------------------------------------------------------------------
# Category sensor: P(observed_category | true_weather)
# Rows = true weather (good, marginal, bad)
# Cols = observed category (good, marginal, bad)
# ---------------------------------------------------------------------------
OBS_CATEGORY = np.array([
    [0.70, 0.25, 0.05],   # true good     → mostly reported as good
    [0.20, 0.60, 0.20],   # true marginal → often reported correctly, some error
    [0.05, 0.25, 0.70],   # true bad      → mostly reported as bad
])

# ---------------------------------------------------------------------------
# Confidence sensor: P(high_confidence | true_weather)
# Intuition: clear good/bad weather is easy to detect → high confidence.
# Marginal weather is ambiguous → sensors disagree → lower confidence.
# ---------------------------------------------------------------------------
P_HIGH_CONF = np.array([0.85, 0.40, 0.80])   # [good, marginal, bad]

N_OBS = 6
OBS_LABELS = [
    'good / low-conf',
    'good / high-conf',
    'marginal / low-conf',
    'marginal / high-conf',
    'bad / low-conf',
    'bad / high-conf',
]

# ---------------------------------------------------------------------------
# Full observation model: OBS_MODEL[w, o] = P(observation o | true weather w)
# Shape: (N_WEATHER, N_OBS)
# Each row sums to 1.
# ---------------------------------------------------------------------------
def _build_obs_model():
    M = np.zeros((N_WEATHER, N_OBS))
    for w in range(N_WEATHER):
        for cat in range(3):
            p_cat = OBS_CATEGORY[w, cat]
            M[w, 2 * cat + 0] = p_cat * (1.0 - P_HIGH_CONF[w])  # low conf
            M[w, 2 * cat + 1] = p_cat * P_HIGH_CONF[w]           # high conf
    return M

OBS_MODEL = _build_obs_model()   # shape (3, 6)


# ---------------------------------------------------------------------------
# Belief operations
# ---------------------------------------------------------------------------

def predict_belief(b):
    """
    Prediction step: propagate belief through the weather transition.

    b_pred(w') = Σ_w  T(w' | w) * b(w)

    WEATHER_TRANSITION[w, w'] = T(w' | w), so:
        b_pred = WEATHER_TRANSITION.T @ b
    """
    return WEATHER_TRANSITION.T @ b


def belief_update(b_pred, obs_idx):
    """
    Update step: reweight predicted belief by likelihood of observation.

    b_new(w') ∝ O(obs | w') * b_pred(w')

    Parameters
    ----------
    b_pred  : predicted belief (after predict_belief), shape (N_WEATHER,)
    obs_idx : index into OBS_LABELS

    Returns
    -------
    b_new : normalised updated belief, shape (N_WEATHER,)
    """
    b_new = OBS_MODEL[:, obs_idx] * b_pred
    norm = b_new.sum()
    return b_new / norm if norm > 1e-12 else b_pred


def obs_probability(b_pred, obs_idx):
    """P(obs | b_pred) = Σ_{w'} O(obs | w') * b_pred(w')"""
    return float(OBS_MODEL[:, obs_idx] @ b_pred)
