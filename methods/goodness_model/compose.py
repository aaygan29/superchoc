"""Compose novel molecule combos that the goodness model predicts will be good.

Criterion 2 in miniature. Given a trained GoodnessModel and a molecule pool, search for
k-molecule combos whose (mean-pooled) mixture embedding maximizes predicted goodness,
subject to a NOVELTY constraint: the combo's mixture embedding must be far enough from a
reference "chocolate" combo, so we are proposing new flavors rather than rediscovering
the reference.

Search is a simple evolutionary hill-climb over combos (swap one member, keep if the
predicted score improves and novelty holds). No dependencies beyond numpy + the model.

Aggregation caveat: mean-pooling assumes additive mixtures. Real olfactory mixtures are
non-additive at the single-neuron level (suppression/synergy; Duchamp-Viret et al. 2003),
so on real data this aggregator must be replaced by a learned mixture model.
"""

from __future__ import annotations

import numpy as np


def _cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(a @ b / (na * nb))


def compose_combos(space, model, k=4, n_combos=5, novelty_min=0.2,
                   ref_embedding=None, iters=300, seed=0):
    """Return ``n_combos`` novel, high-predicted-goodness combos.

    space: FlavorSpace (provides pool + mixture_embedding)
    model: trained GoodnessModel (.predict on an embedding)
    novelty_min: minimum cosine distance from ref_embedding (the chocolate combo)
    """
    rng = np.random.default_rng(seed)
    pool = space.pool
    n = pool.shape[0]

    def score(combo):
        emb = space.mixture_embedding(combo)
        if ref_embedding is not None and _cosine_dist(emb, ref_embedding) < novelty_min:
            return -np.inf, emb
        return float(model.predict(emb)[0]), emb

    found = []
    for _ in range(n_combos):
        combo = tuple(rng.choice(n, size=k, replace=False))
        best_score, _ = score(combo)
        # restart until we have a novelty-valid starting point
        tries = 0
        while best_score == -np.inf and tries < 50:
            combo = tuple(rng.choice(n, size=k, replace=False))
            best_score, _ = score(combo)
            tries += 1
        for _ in range(iters):
            j = rng.integers(k)
            cand = list(combo)
            cand[j] = int(rng.integers(n))
            if len(set(cand)) < k:
                continue
            cand = tuple(cand)
            s, _ = score(cand)
            if s > best_score:
                combo, best_score = cand, s
        emb = space.mixture_embedding(combo)
        found.append({
            "combo": combo,
            "predicted_goodness": best_score,
            "true_goodness": space.true_mixture(combo),   # oracle, for honest scoring only
            "novelty_from_ref": (_cosine_dist(emb, ref_embedding)
                                 if ref_embedding is not None else None),
        })
    return found


def random_combos(space, k=4, n_combos=200, seed=0):
    """Baseline: random k-molecule combos, with their true goodness."""
    rng = np.random.default_rng(seed)
    n = space.pool.shape[0]
    out = []
    for _ in range(n_combos):
        combo = tuple(rng.choice(n, size=k, replace=False))
        out.append({"combo": combo, "true_goodness": space.true_mixture(combo)})
    return out
