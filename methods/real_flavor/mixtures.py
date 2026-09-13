"""Combo (recipe) science: represent, compare, optimize, and stress-test flavor MIXTURES.

superchoc elsewhere improves single molecules and blends them by class composition. But the
conjecture is about COMBOS: chocolate is a recipe, and a new *recipe* of known molecules
(vanillin + mint -> creamy peppermint) can be novel and good even when its parts are not. This
module adds the mixture-level machinery, grounded in flavor-chemistry literature and, where
possible, validated on real human mixture data rather than asserted.

Externally-validated core (validate_snitz):
  Snitz et al. 2013 (PLoS Comput Biol) and Ravia et al. 2020 (Nature) showed a mixture's
  perceptual quality is predicted by the ANGLE between the mixtures' physicochemical-descriptor
  vectors. We reimplement that method (mixture = mean of z-scored physicochemical descriptors;
  perceptual distance = angle in radians) and validate it against Snitz's 360 human-rated
  mixture-similarity pairs. NOTE: we use RDKit physicochemical descriptors as a documented
  substitute for the original Dragon feature set, so the correlation we report is our
  reimplementation's, not the paper's exact number.

Optimal levels (exact math, no perceptual claim):
  Scheffe simplex-centroid mixture design + response-surface fit optimizes the PROPORTIONS of a
  fixed component set under the sum-to-one constraint (the standard food-formulation method).

Hypothesis layer (labeled; rides on the goodness/percept model, which is only synthetic-oracle
validated for non-additivity, so treat as hypotheses):
  - omission_test: drop each component, measure the predicted-percept shift (the in-silico
    analog of aroma-recombination omission testing) -> which components are key / emergent.
  - synergy_score: predicted goodness minus the additive expectation (Bliss-style). Positive =
    hyperadditive "emergent" combo. Ties to FlavorMath.lean (additive <= best component).

Guardrail:
  - olfactory_white_risk: Weiss & Sobel 2012 (PNAS) showed ~30+ equal-intensity components
    converge to an indistinct "olfactory white"; Laing showed humans resolve at most ~3-4
    components. So more components is not more novel. This penalizes over-complex recipes.

Refs: Snitz 2013 doi:10.1371/journal.pcbi.1003184; Ravia 2020 doi:10.1038/s41586-020-2891-7;
Weiss/Sobel 2012 doi:10.1073/pnas.1208110109; Grosch 2001 (OAV/omission) Chem Senses 26:533;
Scheffe 1958 (mixture designs).
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from featurize import smiles_to_ecfp4  # noqa: F401  (kept for downstream callers)

# Physicochemical / topological descriptors (RDKit substitute for Snitz's Dragon set).
_DESC = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Crippen.MolLogP),
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("HBA", Lipinski.NumHAcceptors),
    ("HBD", Lipinski.NumHDonors),
    ("RotB", Lipinski.NumRotatableBonds),
    ("AromRings", rdMolDescriptors.CalcNumAromaticRings),
    ("FracCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("Hetero", Lipinski.NumHeteroatoms),
    ("Rings", rdMolDescriptors.CalcNumRings),
    ("LabuteASA", rdMolDescriptors.CalcLabuteASA),
    ("BertzCT", Descriptors.BertzCT),
    ("MolMR", Crippen.MolMR),
]


@functools.lru_cache(maxsize=20000)
def _descriptors(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    out = []
    for _, fn in _DESC:
        try:
            out.append(float(fn(m)))
        except Exception:
            return None
    return np.array(out, dtype=float)


class MixtureSpace:
    """Physicochemical mixture space with the Snitz/Ravia angle distance."""

    def __init__(self, reference_smiles):
        rows = [d for d in (_descriptors(s) for s in reference_smiles) if d is not None]
        M = np.vstack(rows)
        self.mean = M.mean(0)
        self.std = M.std(0) + 1e-9

    def _z(self, smiles):
        d = _descriptors(smiles)
        return None if d is None else (d - self.mean) / self.std

    def vector(self, smiles_list, weights=None):
        """Mixture vector = (optionally proportion-weighted) mean of z-scored component
        descriptors (Snitz 2013). Weights let perceived position depend on how much of each
        component is present, consistent with concentration/OAV-driven contribution; equal
        weights recover the plain Snitz mean."""
        pairs = [(self._z(s), (weights[i] if weights is not None else 1.0))
                 for i, s in enumerate(smiles_list)]
        pairs = [(z, w) for z, w in pairs if z is not None]
        if not pairs:
            return None
        Z = np.vstack([z for z, _ in pairs])
        w = np.array([w for _, w in pairs], dtype=float)
        w = w / (w.sum() + 1e-12)
        return (w[:, None] * Z).sum(0)

    def distance(self, mix_a, mix_b, weights_a=None, weights_b=None):
        """Perceptual distance = angle (radians) between the two mixture vectors."""
        va, vb = self.vector(mix_a, weights_a), self.vector(mix_b, weights_b)
        if va is None or vb is None:
            return None
        cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        return float(np.arccos(np.clip(cos, -1.0, 1.0)))


# --------------------------------------------------------------------------------------------
# Externally-validated check: does the angle distance predict Snitz's human similarity ratings?
# --------------------------------------------------------------------------------------------
def validate_snitz():
    import pyrfume
    from scipy.stats import spearmanr, pearsonr

    mol = pyrfume.load_data("snitz_2013/molecules.csv")
    cid2smi = {int(c): r["IsomericSMILES"] for c, r in mol.iterrows()
               if isinstance(r["IsomericSMILES"], str)}
    beh = pyrfume.load_data("snitz_2013/behavior.csv")

    ref = [s for s in cid2smi.values() if Chem.MolFromSmiles(s) is not None]
    space = MixtureSpace(ref)

    def cids_to_smiles(cell):
        out = []
        for tok in str(cell).split(","):
            tok = tok.strip()
            if tok.isdigit() and int(tok) in cid2smi:
                out.append(cid2smi[int(tok)])
        return out

    dist, sim = [], []
    for _, r in beh.iterrows():
        A, B = cids_to_smiles(r["StimulusA"]), cids_to_smiles(r["StimulusB"])
        if not A or not B:
            continue
        d = space.distance(A, B)
        if d is None:
            continue
        dist.append(d)
        sim.append(float(r["Similarity"]))
    dist, sim = np.array(dist), np.array(sim)
    # human Similarity is high when mixtures smell alike; angle distance should be LARGE when
    # similarity is LOW -> expect negative correlation.
    sp = float(spearmanr(dist, sim).correlation)
    pr = float(pearsonr(dist, sim)[0])
    return {"n_pairs": int(len(dist)), "spearman_dist_vs_similarity": round(sp, 3),
            "pearson_dist_vs_similarity": round(pr, 3),
            "reading": ("angle distance tracks human dissimilarity (negative corr as expected)"
                        if sp < 0 else "no expected negative relationship (method fails here)")}


# --------------------------------------------------------------------------------------------
# Optimal LEVELS: Odor Activity Value (the flavor-chemistry standard for concentrations).
# OAV = concentration / detection threshold. A molecule contributes when OAV > 1; aroma
# recombination mixes key odorants at OAV-weighted (natural) levels. We implement the machinery
# and REQUIRE measured thresholds as input; we do not ship invented threshold values.
# --------------------------------------------------------------------------------------------
def odor_activity_value(concentration, threshold):
    """OAV = concentration / detection threshold (same units). Both must be measured."""
    if threshold is None or threshold <= 0:
        return None
    return float(concentration) / float(threshold)


def oav_weights(components, concentrations, thresholds):
    """Perceptual weights proportional to OAV (log1p-compressed, hypoadditivity-aware).

    components: list of smiles; concentrations/thresholds: dicts smiles->value (measured).
    Molecules with OAV<=1 (below threshold) get zero weight. Returns normalized weights or None
    if thresholds are missing (we refuse to guess)."""
    oavs = []
    for s in components:
        c, t = concentrations.get(s), thresholds.get(s)
        if c is None or t is None:
            return None
        v = odor_activity_value(c, t)
        oavs.append(max(0.0, (v or 0.0)))
    oavs = np.array(oavs)
    contrib = np.log1p(np.where(oavs > 1.0, oavs, 0.0))  # >threshold only, compressed
    tot = contrib.sum()
    return (contrib / tot) if tot > 0 else None


# --------------------------------------------------------------------------------------------
# Optimal PROPORTIONS: Scheffe simplex-centroid mixture design + response-surface fit. Pure
# math (no perceptual claim). NOTE: an ADDITIVE response optimizes to a single component (see
# FlavorMath.lean best-part bound), so ratio optimization is only non-trivial for a
# non-additive / interaction-bearing response - which is exactly why synergy matters.
# --------------------------------------------------------------------------------------------
def simplex_centroid(k):
    """Scheffe simplex-centroid design points for k components (all equal-split subsets)."""
    pts = []
    for r in range(1, k + 1):
        for combo in itertools.combinations(range(k), r):
            p = np.zeros(k)
            p[list(combo)] = 1.0 / r
            pts.append(p)
    return np.array(pts)


def simplex_lattice_deg2(k):
    """Degree-2 simplex-lattice: pure vertices e_i plus binary 50/50 edge midpoints. Matches a
    quadratic Scheffe model and stays tractable for large k (k + C(k,2) points, vs 2^k-1 for
    the full centroid)."""
    pts = []
    for i in range(k):
        p = np.zeros(k); p[i] = 1.0; pts.append(p)
    for i in range(k):
        for j in range(i + 1, k):
            p = np.zeros(k); p[i] = p[j] = 0.5; pts.append(p)
    return np.array(pts)


def _scheffe_features(P):
    """Scheffe quadratic: linear terms x_i plus cross terms x_i*x_j (no intercept)."""
    k = P.shape[1]
    cols = [P[:, i] for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            cols.append(P[:, i] * P[:, j])
    return np.vstack(cols).T


def optimize_ratios(components, score_fn, n_grid=6000, seed=0):
    """Fit a Scheffe response surface to score_fn over a simplex-centroid design, then search
    the simplex for the proportions that maximize the fitted response.

    score_fn(proportions: np.ndarray) -> float  (proportions sum to 1, len == len(components)).
    Returns best proportions, fitted vs realized values, and the design R^2 (fit quality)."""
    k = len(components)
    P = simplex_centroid(k) if k <= 5 else simplex_lattice_deg2(k)
    y = np.array([float(score_fn(p)) for p in P])
    X = _scheffe_features(P)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    # search the simplex (Dirichlet samples + the design points) on the fitted surface
    rng = np.random.default_rng(seed)
    cand = np.vstack([P, rng.dirichlet(np.ones(k), size=n_grid)])
    fitted = _scheffe_features(cand) @ beta
    best = int(np.argmax(fitted))
    bp = cand[best]
    interior = bool(bp.max() < 0.95)  # <0.95 on any single component => a real blend
    return {"components": components,
            "best_proportions": [round(float(x), 3) for x in bp],
            "predicted_response": round(float(fitted[best]), 4),
            "design_r2": round(r2, 3), "n_design_points": int(len(P)),
            "interior_optimum": interior,
            "reading": ("interior blend optimum (interaction present)" if interior else
                        "optimum collapses to a single component: the response is effectively "
                        "additive (FlavorMath.lean best-part bound). A non-degenerate ratio "
                        "optimum requires a validated non-additive/synergy response, which is "
                        "not yet available for real molecules.")}


# --------------------------------------------------------------------------------------------
# HYPOTHESIS layer (rides on a percept model; label outputs as hypotheses, not measurements).
# --------------------------------------------------------------------------------------------
def omission_test(components, combo_predict_fn):
    """In-silico aroma-recombination omission: drop each component, measure the predicted
    change. combo_predict_fn(list_of_smiles) -> scalar or vector. Returns per-component impact,
    ranked (largest impact = most 'key'). A component whose omission strongly changes the
    percept is essential; the full-vs-leave-one-out deltas hypothesize emergent structure."""
    full = np.atleast_1d(np.asarray(combo_predict_fn(components), dtype=float))
    out = []
    for i, s in enumerate(components):
        rest = components[:i] + components[i + 1:]
        if not rest:
            continue
        left = np.atleast_1d(np.asarray(combo_predict_fn(rest), dtype=float))
        impact = float(np.linalg.norm(full - left))
        out.append({"omitted": s, "impact": round(impact, 4)})
    out.sort(key=lambda d: d["impact"], reverse=True)
    return out


def synergy_score(components, combo_predict_fn, mono_predict_fn, weights=None):
    """Bliss-style synergy: predicted combo goodness minus the additive expectation.

    combo_predict_fn(list) -> scalar (non-additive prediction); mono_predict_fn(smiles) ->
    scalar (single-molecule goodness). additive expectation = weighted mean of monos (the
    FlavorMath.lean bound: additive <= best component). synergy > 0 => hyperadditive/emergent."""
    k = len(components)
    w = np.ones(k) / k if weights is None else np.asarray(weights, dtype=float)
    monos = np.array([float(mono_predict_fn(s)) for s in components])
    additive = float((w * monos).sum())
    combo = float(combo_predict_fn(components))
    return {"combo_predicted": round(combo, 3), "additive_expectation": round(additive, 3),
            "best_component": round(float(monos.max()), 3),
            "synergy": round(combo - additive, 3),
            "exceeds_best_component": bool(combo > monos.max()),
            "reading": ("hyperadditive (emergent): combo beats additive expectation"
                        if combo > additive else "hypoadditive/additive: no emergent gain")}


# --------------------------------------------------------------------------------------------
# Guardrail: olfactory white (Weiss & Sobel 2012). Many equal components -> indistinct percept.
# --------------------------------------------------------------------------------------------
def olfactory_white_risk(n_components):
    """Risk that a recipe collapses toward 'olfactory white' (indistinct). ~<=4 components are
    perceptually resolvable (Laing); risk ramps up toward ~30 (Weiss & Sobel)."""
    if n_components <= 4:
        return {"n_components": n_components, "white_risk": 0.0, "reading": "resolvable (configural)"}
    risk = min(1.0, (n_components - 4) / (30 - 4))
    return {"n_components": n_components, "white_risk": round(risk, 3),
            "reading": ("approaching olfactory white (indistinct)" if risk > 0.6
                        else "moderately complex")}


def validate_ravia():
    """Second external validation on Ravia et al. 2020 (Nature): 195 pairwise mixture-similarity
    ratings (behavior_2). Same method as Snitz: correlate the angle distance with rated
    similarity; expect a negative relationship."""
    import re
    import pyrfume
    from scipy.stats import spearmanr, pearsonr

    st = pyrfume.load_data("ravia_2020/stimuli.csv")
    mol = pyrfume.load_data("ravia_2020/molecules.csv")
    cid2smi = {int(c): r["IsomericSMILES"] for c, r in mol.iterrows()
               if isinstance(r["IsomericSMILES"], str)}
    b2 = pyrfume.load_data("ravia_2020/behavior_2.csv").reset_index()

    def stim_smiles(stim_id):
        if stim_id not in st.index:
            return []
        cids = re.split(r"[;,]", str(st.loc[stim_id, "CID"]))
        out = []
        for t in cids:
            t = t.strip()
            if t.isdigit() and int(t) in cid2smi:
                s = cid2smi[int(t)]
                if Chem.MolFromSmiles(s) is not None:
                    out.append(s)
        return out

    ref = [s for s in cid2smi.values() if Chem.MolFromSmiles(s) is not None]
    space = MixtureSpace(ref)
    dist, sim = [], []
    for _, r in b2.iterrows():
        A, B = stim_smiles(r["Stimulus 1"]), stim_smiles(r["Stimulus 2"])
        if not A or not B:
            continue
        d = space.distance(A, B)
        if d is None:
            continue
        dist.append(d); sim.append(float(r["RatedSimilarity"]))
    dist, sim = np.array(dist), np.array(sim)
    sp = float(spearmanr(dist, sim).correlation)
    pr = float(pearsonr(dist, sim)[0])
    return {"n_pairs": int(len(dist)), "spearman_dist_vs_similarity": round(sp, 3),
            "pearson_dist_vs_similarity": round(pr, 3),
            "reading": ("angle distance tracks human dissimilarity (negative corr as expected)"
                        if sp < 0 else "no expected negative relationship (method fails here)")}


if __name__ == "__main__":
    import json
    print("Snitz 2013 validation:", json.dumps(validate_snitz(), indent=2))
    print("Ravia 2020 validation:", json.dumps(validate_ravia(), indent=2))
    # small demos of the exact-math pieces (no external calls)
    demo = ["O=Cc1ccc(O)c(OC)c1", "CC(C)C1CCC(C)CC1O", "CC(C)=CCCC(C)(O)C=C"]  # vanillin,menthol,linalool
    ref = ["CCCC(=O)OCC", "CC(C)CCOC(C)=O", "O=Cc1ccccc1"]
    # build the space on a real reference set (Snitz molecules) so the angle is meaningful
    import pyrfume as _pf
    _m = _pf.load_data("snitz_2013/molecules.csv")
    _ref = [r["IsomericSMILES"] for _, r in _m.iterrows() if isinstance(r["IsomericSMILES"], str)]
    space = MixtureSpace(_ref)
    print("perceptual distance (radians) demo vs ref:", round(space.distance(demo, ref), 3))
    print("olfactory white risk (10 comps):", json.dumps(olfactory_white_risk(10)))
    # ratio optimization needs a non-additive response to be non-degenerate:
    def toy_nonadditive(p):
        return float(p[0]*60 + p[1]*55 + p[2]*50 + 40*p[0]*p[1])  # vanillin x menthol interaction
    print("optimize_ratios (toy non-additive):", json.dumps(optimize_ratios(demo, toy_nonadditive)))

