# Pilot Decision-Making Under Uncertainty

*Applying algorithmic decision theory to aviation — from everyday go/no-go calls to the edge of the flight envelope.*

> Based on *Algorithms for Decision Making* — Kochenderfer, Wheeler & Wray

---

## Why This Matters

Pilots make consequential decisions constantly, almost never with complete information. The weather is ambiguous. The aircraft is behaving slightly differently than expected. ATC just changed your routing. Fuel math gets tight.

Most of the time, pilots handle this well — through training, heuristics, and experience. But accidents happen. And they almost always trace back not to ignorance, but to how uncertainty was *managed*.

Decision theory gives us a formal language for what's really going on. This project applies that language to aviation.

---

## Part 1 — The Structure of Pilot Uncertainty

Not all uncertainty is the same, and the textbook outlines two types:

**Aleatoric uncertainty**: irreducible randomness. The turbulence over the mountains might be moderate or severe; no amount of additional information tells you which until you're in it. You can only reason about probability.

**Epistemic uncertainty**: uncertainty from incomplete knowledge. You don't know if the SIGMET was updated in the last hour. You don't know if the MEF on the sectional reflects recent construction. This type *can* be reduced by gathering more information.

The distinction matters practically. Waiting for more information helps with epistemic uncertainty. It doesn't help with aleatoric uncertainty and wasting time trying to resolve it can burn fuel and options.

### The Three Domains of Pilot Uncertainty

| Domain | What's uncertain | Example |
|---|---|---|
| **State** | Where you are / what the environment is doing | Weather between you and the destination |
| **Model** | How the aircraft or system will behave | Will this engine keep running at high altitude? |
| **Outcome** | What happens if you take action X | If I descend now, will I break out before the terrain? |

Most in-flight emergencies involve all three simultaneously.

---

## Part 2 — Frameworks from Decision Theory

### 2.1 Expected Utility Theory — The Rational Pilot

The foundation of formal decision-making: choose the action that maximizes **expected utility**.

```
EU(action) = Σ P(outcome | action) × U(outcome)
```

For a go/no-go decision:
- `P(outcome | action)` = probability of each possible outcome (safe arrival, weather encounter, divert, etc.)
- `U(outcome)` = how much you value each outcome (not just survival — also mission completion, passenger inconvenience, etc.)

**The critical insight:** pilots aren't expected-value maximizers — they're expected-*utility* maximizers. Utility functions are nonlinear. A 10% chance of a fatal accident isn't "just" 10 times worse than a 1% chance — it's catastrophically worse. This is why trained pilots are correctly risk-averse in ways that look overcautious to outsiders.

The practical implication: when stakes are asymmetric (one bad outcome is catastrophic), you should *not* make the statistically "average" choice. You should make the choice that eliminates the tail risk.

---

### 2.2 Bayesian Updating — The Mental Model in Flight

A pilot's mental model is a probability distribution, not a single belief. You start a flight with a prior estimate of what the weather will do. As you fly, you get new information (PIREPS, ATC advisories, what you see out the window) and update.

Formally:

```
P(state | observation) ∝ P(observation | state) × P(state)
```

**Posterior** (updated belief) ∝ **Likelihood** (how probable is what I'm seeing, given the state?) × **Prior** (what did I believe before?)

#### Why This Breaks Down in Real Flight

Three failure modes that accident reports echo constantly:

1. **Confirmation bias** — overweighting evidence that supports your prior ("the weather will be fine") and discounting contradictory signals (the building cumulus to the west).

2. **Anchoring** — failing to update enough when new evidence arrives. The pilot who planned for VFR and encounters marginal IMC still thinks of themselves as "in a VFR scenario."

3. **Availability heuristic** — overweighting dramatic outcomes (a crash you heard about) or underweighting scenarios outside your experience (spatial disorientation if you've never had it).

Bayesian reasoning isn't just math — it's the discipline of *actually updating* when evidence arrives.

---

### 2.3 Sequential Decisions — Go/No-Go Is a Policy, Not a Moment

The go/no-go decision isn't a single binary call made at preflight. It's a **policy**: a function that maps every possible state of the world to an action at each moment in the flight.

This is where Markov Decision Processes (MDPs) come in.

An MDP models the problem as:
- **States** — your current situation (fuel state, weather conditions, distance to destination, etc.)
- **Actions** — what you can do (continue, divert, declare emergency, descend, etc.)
- **Transition function** — how likely each action is to land you in each new state
- **Reward function** — what outcomes you're optimizing for

The solution is a **policy** — a mapping from every state to the best action. The pilot who has genuinely internalized good decision-making has essentially learned an approximation of this policy.

**Key insight:** the decision to divert isn't made "when things get bad." It's made *before the flight*, in the form of decision criteria. "If I'm not VFR by X waypoint, I'm turning around" is a policy commitment. Making this explicit in advance removes it from the emotional, high-workload moment when it's hardest to think clearly.

---

### 2.4 Partial Observability — You Can't See Everything

Real flight is not an MDP — it's a **POMDP** (Partially Observable Markov Decision Process). You don't observe the true state of the world directly. You observe *evidence*, and you maintain a **belief state**, which is a probability distribution over possible true states.

```
Belief state b(s) = P(true state is s | all observations so far)
```

Classic aviation examples:
- **Icing:** You see no ice accumulation yet, but is that because there's no ice in the clouds ahead or because you just entered them?
- **Engine health:** The gauges read normal, but is the engine actually healthy or is there a developing failure that hasn't manifested yet?
- **Spatial disorientation:** Your instruments say one thing; your vestibular system says another. Which is the true state?

The POMDP framework says: don't try to collapse the uncertainty to a single "most likely" state. Maintain the full belief distribution, and choose actions that perform well *across* that distribution — especially avoiding catastrophic outcomes in low-probability but high-consequence states.

This is why "trust your instruments" is such fundamental training — it's teaching pilots to use the correct observation model rather than their unreliable body-state estimates.

---

## Part 3 — Classic Scenarios Through the Lens

### Weather Go/No-Go

- **State uncertainty:** You don't know what conditions are actually like at the destination.
- **Policy vs. moment:** Establish decision points before departure — "if X, I turn around" — not in the air under stress.
- **Bayesian update:** PIREPs, METARs, and what you see out the window are likelihood functions. Update your belief; don't anchor to the forecast.
- **Asymmetric utility:** The cost of canceling a VFR flight is low. The cost of flying into IMC unprepared is catastrophically high. The threshold for action should reflect this asymmetry.

### Fuel Management

- **Model uncertainty:** You planned for X burn rate. Actual burn is higher due to headwinds. At what point does your model need updating?
- **Sequential decisions:** "Minimum fuel" declarations are a late-stage artifact of poor policy design earlier in the flight.
- **Expected utility:** The expected cost of landing short with extra fuel is small. The expected cost of running dry is catastrophic. Rational pilots carry more fuel than the average scenario requires.

### Mechanical Anomaly

- **Partial observability:** A single unusual gauge reading could be an instrument error or a real developing failure. You can't observe the true state — only symptoms.
- **Belief updating:** As symptoms evolve or new readings confirm/deny, update.
- **Policy under uncertainty:** The correct action is often the one that preserves options — land sooner than necessary rather than press on and discover it was real.

---

## Part 4 — Flight Test: Uncertainty at the Frontier

General aviation operates within a thoroughly validated envelope. Flight test operates *at and beyond* its edges — which makes decision theory not just useful but essential.

### The Build-Up Approach

Flight test engineers don't jump to the edge of the envelope. They use **incremental exploration**: expanding the tested region step by step, gathering data at each point before proceeding.

This is sequential decision-making under model uncertainty. At each point, you're asking: *does our model of the aircraft's behavior remain accurate here?* If yes, proceed. If the data deviates, stop and understand why before continuing.

The cost of not doing this is the accident. The Grover Loening Award exists because people have died finding where the envelope actually ends.

### Quantifying the Unknown

In flight test, uncertainty is explicitly modeled:
- **Parameter uncertainty**: we know the structure of the model but not its exact coefficients. More test points → tighter confidence intervals.
- **Structural uncertainty**: our model of the aerodynamics may be wrong in this regime entirely. No amount of refinement of current data resolves this; you need new data from new flight conditions.
- **Measurement uncertainty**: the instrumentation introduces noise. Test instrumentation exists to minimize this.

The test plan exists to navigate this in a controlled way by designing experiments that maximally reduce uncertainty in the most dangerous unknowns first.

### The Test Pilot's Belief State

A test pilot maintaining situational awareness during an envelope expansion flight is, in a real sense, running a live POMDP. Their belief state incorporates:
- Aircraft structural margins (partially observable via feel, vibration, data stream)
- Aerodynamic behavior (observed but model-uncertain)
- Systems health (partially observable via gauges + anomalies)
- Chase aircraft / control room information (noisy but valuable observations)

The decision to continue or knock it off is a policy over this belief state, with a very asymmetric utility function.

---

## Takeaways

1. **Uncertainty is the job**: not a nuisance to eliminate, but the fundamental condition aviation operates in. The question isn't how to get certainty; it's how to make good decisions without it.

2. **Decisions are policies, not moments**: the best time to make a hard in-flight call is before the flight, as a conditional rule. Under stress and high workload, explicit pre-committed policies outperform in-the-moment reasoning.

3. **Bayesian discipline**: update your mental model when evidence arrives. The most dangerous pilots aren't the ones who don't know things; they're the ones who don't update when they should.

4. **Asymmetric stakes demand asymmetric thresholds**: rational behavior under catastrophic downside risk means accepting a higher rate of "unnecessary" conservative actions. The cost of the false alarm is always less than the cost of the miss.

5. **Flight test is the extremum**: everything above applies, amplified. Test operations make uncertainty explicit, quantitative, and the center of the mission.

