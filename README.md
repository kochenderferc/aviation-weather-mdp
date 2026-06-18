# Aviation Weather Go/No-Go Decision — MDP with Value Iteration

Can we find the mathematically optimal policy for a pilot deciding whether to continue a flight or divert when weather deteriorates?

This project answers that question using a **Markov Decision Process (MDP)** solved with **value iteration**: a core algorithm in sequential decision making under uncertainty. See [my notes on pilot decision making](background/pilot-decision-making-uncertainty.md) for some context.

---

## The Problem

A pilot is flying from Phoenix (KPHX) to San Diego (KSAN) via two intermediate waypoints. At each waypoint, the pilot observes current weather conditions (**good**, **marginal**, or **bad**) and must decide:

- **Continue** to the next waypoint, or
- **Divert** to the nearest alternate airport

Weather evolves stochastically between waypoints. The MDP finds the optimal policy: the action that maximizes expected long-term reward at every possible state.

---

## MDP Formulation

| Component | Definition |
|-----------|------------|
| **States** | `(waypoint, weather)` pairs + terminal states `arrived` / `diverted` |
| **Actions** | `continue`, `divert` |
| **Transitions** | Weather evolves per a 3×3 stochastic matrix; continuing advances to the next waypoint |
| **Rewards** | Arrive: +100, Divert: −20, Continue (good/marginal/bad): +2 / −5 / −80 |
| **Discount γ** | 0.95 |

Weather transition matrix (row = current, col = next):

|          | → good | → marginal | → bad |
|----------|--------|------------|-------|
| good     | 0.70   | 0.25       | 0.05  |
| marginal | 0.30   | 0.40       | 0.30  |
| bad      | 0.05   | 0.25       | 0.70  |

---

## Algorithm: Value Iteration

Value iteration applies the **Bellman backup** repeatedly until the value function converges:

$$U_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} T(s'|s,a)\, U_k(s') \right]$$

Convergence is guaranteed because the Bellman backup is a contraction mapping. The algorithm stops when the Bellman residual $\|U_{k+1} - U_k\|_\infty$ falls below a tolerance of $10^{-6}$.

---

## Results

The optimal policy converges in **5 iterations**:

| State | Optimal Action | Value |
|-------|---------------|-------|
| (KPHX, good) | continue | 63.15 |
| (KPHX, marginal) | continue | 27.30 |
| (KPHX, bad) | **divert** | −20.00 |
| (KBXK, good) | continue | 78.77 |
| (KBXK, marginal) | continue | 40.93 |
| (KBXK, bad) | **divert** | −20.00 |
| (KTRM, good) | continue | 93.34 |
| (KTRM, marginal) | continue | 65.87 |
| (KTRM, bad) | **divert** | −20.00 |
| (KSAN, good) | continue | 102.00 |
| (KSAN, marginal) | continue | 95.00 |
| (KSAN, bad) | continue | 20.00 |

**Key insight:** At KSAN (the final waypoint), even bad weather yields a "continue" decision — arrival (+100) plus the bad-weather penalty (−80) = +20, which beats diverting (−20). Earlier waypoints with bad weather always trigger a divert because the pilot would have to fly through multiple bad-weather legs.

---

## Visualizations

| Plot | Description |
|------|-------------|
| `plots/value_function.png` | Heatmap of U*(s) over all states |
| `plots/policy_map.png` | Optimal action at each (waypoint, weather) state |
| `plots/convergence.png` | Bellman residual per iteration |
| `plots/simulation_*.png` | Simulated flights from each starting weather condition |

---

## Project Structure

```
aviation-weather-mdp/
├── scenario.py          — route, weather model, rewards, discount factor
├── mdp.py               — state space, transition function, reward function
├── value_iteration.py   — Bellman backup and value iteration algorithm
├── visualize.py         — all plotting functions + flight simulation
├── main.py              — run everything, print results, save plots
├── notebook.ipynb       — full walkthrough with math, code, and interpretation
└── plots/               — saved visualizations
```

---

## Usage

```bash
pip install numpy matplotlib
python main.py
```

Or open `notebook.ipynb` for a full narrative walkthrough.

---

## Background

This project implements concepts from:

> Kochenderfer, Wheeler & Wray. *Algorithms for Decision Making*, Ch. 7: Exact Solution Methods. MIT Press, 2022. [(free PDF)](https://algorithmsbook.com/decisionmaking/)

Related topics this project connects to:
- **Approximate dynamic programming** (Ch. 8–9) — needed when state spaces are large
- **Reinforcement learning** (Ch. 17) — when the transition model is unknown
- **POMDPs** (Ch. 19–22) — when the pilot can't directly observe weather (only sense it)
- **Multiagent MDPs** (Ch. 24–27) — multiple aircraft sharing airspace
