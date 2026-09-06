# Online active flavor search

A method for the expensive question in Wintermute's Conjecture 4: flavor space is huge,
human panels are cheap-but-not-free, so **which molecule do you taste next?** This is an
online (sequential) experimental-design loop: it keeps a model of deliciousness over the
flavor embedding and picks each next query to make the most progress per panel cycle.
This is the lever behind Criterion 2's "cheaper and safer design cycles".

## The method

1. Lift each molecule's flavor embedding into **Random Fourier Features** (Rahimi & Recht,
   2007), so a closed-form Bayesian linear model approximates a Gaussian process with an
   RBF kernel while staying cheap to update after every single tasting.
2. Fit a **Bayesian linear model** (posterior mean + variance) online.
3. Pick the next molecule by **Upper Confidence Bound** (mean + beta * sd), never
   re-querying. `beta` is small (0.1) by design (see "What failed", below).

`ground_truth.py` is a reproducible synthetic stand-in for the real target (a POM-style
embedding mapped to a noisy deliciousness surface). It exists only to answer one honest
question: **does the online method beat random screening?** If it cannot beat random on a
controlled problem, it will not help on the real one.

## Validation result (reproducible)

`python validate.py` on 25 seeds, budget 50, against random screening:

| metric | UCB (validated) | random |
|---|---|---|
| mean best-found deliciousness | **0.633** | 0.558 |
| mean gain over random | **+0.074** | - |
| win rate over seeds | **25/25** | - |
| best at HALF budget vs random at FULL budget | **+0.062** | - |

Pre-registered bar (set before the final run): gain >= 0.02, win rate >= 0.75, and
active-at-half-budget >= random-at-full-budget. **UCB passes all three.** Unit tests:
`python test_active_search.py` (6/6).

## What failed, and why it is in the repo

Science, not decoration. Two failure modes were found and are documented so they are not
re-fought:

1. **Flat ground truth.** A first synthetic surface put all deliciousness in a
   vanishing-measure needle (signal std 0.004 vs noise 0.03); no method can beat random
   there. Fixed by making deliciousness vary smoothly and learnably across the pool.
2. **Over-exploration.** With an accurate surrogate (predicted-vs-true corr ~0.95), a
   large UCB exploration bonus (beta=2.0) and **Thompson sampling** both *lose* to random:
   RFF predictive variance is largest in low-density, low-deliciousness tails, so
   aggressive exploration chases garbage. Near-greedy UCB (beta=0.1) wins decisively.

Consequently **Thompson sampling is NOT validated here** (mean gain -0.041, win rate
4/25). It remains in `active_search.py` for comparison and is clearly marked; do not use
it as the method without re-tuning exploration.

## Files

- `ground_truth.py` - reproducible synthetic flavor space + oracle.
- `active_search.py` - RFF map, Bayesian linear model, UCB/Thompson acquisition, loops.
- `validate.py` - the pre-registered active-vs-random experiment.
- `test_active_search.py` - unit tests.
- `results/validation.json` - the committed result of the run above.

## From synthetic to real

Swap `ground_truth.query(idx)` for a real oracle: the C1 forward flavor model (predicted
pleasantness) for in-silico cycles, or a blinded human panel rating for real ones. The
embedding becomes a POM/GNN embedding of real molecules (see
[../../docs/GROUND_TRUTH.md](../../docs/GROUND_TRUTH.md)). The loop is unchanged.

_Conjecture and framing: Jake Wintermute (see repo README). Method and validation: mine._
