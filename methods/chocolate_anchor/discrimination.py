"""Can humans tell two odor mixtures apart? A psychometric model fitted and tested on real triangle tests.

Data: Bushdid et al. 2014 (Science), via pyrfume 'bushdid_2014'. 264 triangle tests (one odd
mixture, two copies of the other), 26 subjects each. Chance = 1/3. Model:

    p_correct = 1/3 + (2/3) * sigmoid(a + b . x)

fitted by binomial maximum likelihood on per-triangle counts (k correct of 26). Features x:
  overlap     fraction of components shared (Bushdid's own predictor)
  log_n       log number of components per mixture
  structural  Snitz/Ravia angle distance in RDKit physicochemical space (mixtures.MixtureSpace)
  perceptual  angle distance between summed Leffingwell odor-descriptor vectors

Honest scope: see DISCRIMINATION.md. Run this file to regenerate results/discrimination.json
and results/discrimination_model.json.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "real_flavor"))  # real_flavor modules are top-level imports
from mixtures import MixtureSpace  # noqa: E402
from rdkit import Chem  # noqa: E402

RESULTS_DIR = _HERE / "results"
CHANCE = 1.0 / 3.0
N_SUBJECTS = 26
_EPS = 1e-9
RIDGE = 1e-3  # L2 on standardised slopes; keeps the combined fit stable, not tuned

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "overlap_only": ("overlap",),
    "overlap": ("overlap", "log_n"),  # Bushdid's predictors; the baseline everything is compared to
    "structural": ("structural",),
    "perceptual": ("perceptual",),
    "combined": ("overlap", "log_n", "structural", "perceptual"),
}
BASELINE = "overlap"
DEFAULT_MODEL = "combined"


# ------------------------------------------------------------------ chemistry helpers
def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def _flat(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, isomericSmiles=False)


class DescriptorTable:
    """Leffingwell odor descriptors keyed by SMILES (exact canonical first, then stereo-stripped)."""

    def __init__(self) -> None:
        import pyrfume
        beh = pyrfume.load_data("leffingwell/behavior.csv")
        mol = pyrfume.load_data("leffingwell/molecules.csv")
        self.names = list(beh.columns)
        exact: dict[str, np.ndarray] = {}
        flat_lists: dict[str, list[np.ndarray]] = {}
        for cid, row in mol.iterrows():
            smi = row["IsomericSMILES"]
            if not isinstance(smi, str) or int(cid) not in beh.index:
                continue
            if Chem.MolFromSmiles(smi) is None:
                continue
            vec = beh.loc[cid].to_numpy(dtype=float)
            exact[canonical(smi)] = vec
            flat_lists.setdefault(_flat(smi), []).append(vec)
        self._exact = exact
        self._flat = {k: np.mean(v, axis=0) for k, v in flat_lists.items()}

    def get(self, smiles: str) -> np.ndarray | None:
        c = canonical(smiles)
        if c in self._exact:
            return self._exact[c]
        return self._flat.get(_flat(smiles))


_TABLE: DescriptorTable | None = None


def descriptor_table() -> DescriptorTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = DescriptorTable()
    return _TABLE


def angle(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        raise ValueError("zero vector has no angle")
    return float(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0)))


def perceptual_vector(smiles: Sequence[str], weights: Sequence[float] | None = None
                      ) -> tuple[np.ndarray | None, list[str]]:
    """Weighted sum of Leffingwell vectors. Returns (vector or None if nothing covered, missing smiles)."""
    w = [1.0] * len(smiles) if weights is None else list(weights)
    if len(w) != len(smiles):
        raise ValueError("weights and smiles differ in length")
    table = descriptor_table()
    total = np.zeros(len(table.names))
    missing: list[str] = []
    covered = 0
    for s, wi in zip(smiles, w):
        vec = table.get(s)
        if vec is None:
            missing.append(s)
        else:
            total += wi * vec
            covered += 1
    if covered == 0 or not np.any(total):
        return None, missing
    return total, missing


# ------------------------------------------------------------------ Bushdid triangles
@dataclass(frozen=True)
class Triangle:
    stimulus: int
    odd_cids: frozenset[int]
    dup_cids: frozenset[int]
    n_components: int
    n_differ: int
    k_correct: int

    @property
    def overlap(self) -> float:
        return 1.0 - self.n_differ / self.n_components

    @property
    def differing_cids(self) -> frozenset[int]:
        return self.odd_cids ^ self.dup_cids


@dataclass(frozen=True)
class Bushdid:
    triangles: list[Triangle]
    cid_to_smiles: dict[int, str]


def load_bushdid() -> Bushdid:
    """Load and verify the reading of the data; raises on any inconsistency."""
    import pyrfume
    stim = pyrfume.load_data("bushdid_2014/stimuli.csv")
    beh = pyrfume.load_data("bushdid_2014/behavior.csv")
    mols = pyrfume.load_data("bushdid_2014/molecules.csv")
    cid_to_smiles = {int(c): r["IsomericSMILES"] for c, r in mols.iterrows()}
    if any(not isinstance(s, str) or Chem.MolFromSmiles(s) is None for s in cid_to_smiles.values()):
        raise ValueError("bushdid molecules.csv has a missing or unparseable SMILES")
    mol_cols = [f"Molecule {i}" for i in range(1, 31)]
    counts = beh.groupby(level=0)["Correct"].agg(["sum", "size"])
    if not (counts["size"] == N_SUBJECTS).all():
        raise ValueError("every triangle must have exactly 26 subject responses")
    if set(counts.index) != set(stim.index):
        raise ValueError("behavior and stimuli disagree on Stimulus ids")

    def cids(row) -> frozenset[int]:
        vals = [int(v) for v in row[mol_cols].dropna() if int(v) != 0]
        if len(vals) != len(set(vals)):
            raise ValueError("duplicate molecule inside one mixture")
        return frozenset(vals)

    tris: list[Triangle] = []
    for sid, g in stim.groupby(level=0):
        if len(g) != 3 or (g["Answer"] == "right").sum() != 1:
            raise ValueError(f"stimulus {sid}: expected 1 'right' + 2 'wrong' rows")
        odd = cids(g[g["Answer"] == "right"].iloc[0])
        dups = [cids(r) for _, r in g[g["Answer"] == "wrong"].iterrows()]
        if dups[0] != dups[1]:
            raise ValueError(f"stimulus {sid}: the two duplicate vials differ in composition")
        dup = dups[0]
        head = g[g["Answer"] == "right"].iloc[0]
        n, n_diff = int(head["Components in mixtures"]), int(head["Components that differ"])
        if len(odd) != n or len(dup) != n:
            raise ValueError(f"stimulus {sid}: component count mismatch")
        if len(odd - dup) != n_diff or len(dup - odd) != n_diff:
            raise ValueError(f"stimulus {sid}: 'Components that differ' != computed set difference")
        if abs(100.0 * (n - n_diff) / n - float(head["% mixture overlap"])) > 0.5:
            raise ValueError(f"stimulus {sid}: '% mixture overlap' inconsistent")
        if not (odd | dup) <= cid_to_smiles.keys():
            raise ValueError(f"stimulus {sid}: molecule missing from molecules.csv")
        tris.append(Triangle(int(sid), odd, dup, n, n_diff, int(counts.loc[sid, "sum"])))
    return Bushdid(tris, cid_to_smiles)


# ------------------------------------------------------------------ features
def structural_space(reference_smiles: Sequence[str]) -> MixtureSpace:
    return MixtureSpace(list(reference_smiles))


def structural_distance(space: MixtureSpace, a: Sequence[str], b: Sequence[str],
                        wa: Sequence[float] | None = None, wb: Sequence[float] | None = None) -> float:
    d = space.distance(list(a), list(b), wa, wb)
    if d is None:
        raise ValueError("structural distance undefined (no parseable components)")
    return d


def perceptual_distance(a: Sequence[str], b: Sequence[str], wa=None, wb=None
                        ) -> tuple[float | None, list[str]]:
    va, ma = perceptual_vector(a, wa)
    vb, mb = perceptual_vector(b, wb)
    if va is None or vb is None:
        return None, ma + mb
    return angle(va, vb), ma + mb


FEATURES = ("overlap", "log_n", "structural", "perceptual")


@dataclass(frozen=True)
class FeatureTable:
    """Feature matrix over the triangles for which every feature is defined, plus the audit."""
    triangles: list[Triangle]
    X: np.ndarray  # columns = FEATURES
    k: np.ndarray
    audit: dict


def build_features(data: Bushdid) -> FeatureTable:
    smi = data.cid_to_smiles
    space = structural_space(smi.values())
    table = descriptor_table()
    covered = {c for c, s in smi.items() if table.get(s) is not None}
    kept: list[Triangle] = []
    rows: list[list[float]] = []
    n_missing_comp = n_undefined = 0
    for t in data.triangles:
        a = [smi[c] for c in sorted(t.odd_cids)]
        b = [smi[c] for c in sorted(t.dup_cids)]
        if any(c not in covered for c in t.odd_cids | t.dup_cids):
            n_missing_comp += 1
        pd_, _ = perceptual_distance(a, b)
        if pd_ is None:
            n_undefined += 1
            continue
        kept.append(t)
        rows.append([t.overlap, math.log(t.n_components), structural_distance(space, a, b), pd_])
    audit = {
        "leffingwell_molecules_covered": len(covered), "bushdid_molecules": len(smi),
        "triangles_total": len(data.triangles),
        "triangles_with_any_uncovered_component": n_missing_comp,
        "triangles_perceptual_undefined_and_excluded_from_all_models": n_undefined,
        "triangles_used": len(kept),
        "note": ("Components lacking Leffingwell labels are skipped when summing a mixture's vector; "
                 "a triangle is excluded only if a whole mixture has no labelled component. All "
                 "models are scored on the same triangles."),
    }
    return FeatureTable(kept, np.array(rows), np.array([t.k_correct for t in kept]), audit)


# ------------------------------------------------------------------ psychometric model
def p_correct_from_z(z: np.ndarray) -> np.ndarray:
    return CHANCE + (1.0 - CHANCE) * expit(z)


@dataclass(frozen=True)
class DiscriminationModel:
    name: str
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    intercept: float
    slopes: tuple[float, ...]

    def p_correct(self, X: np.ndarray) -> np.ndarray:
        Z = (X - np.array(self.mean)) / np.array(self.scale)
        return p_correct_from_z(self.intercept + (Z * np.array(self.slopes)).sum(1))

    def to_json(self) -> dict:
        return {"name": self.name, "feature_names": list(self.feature_names), "mean": list(self.mean),
                "scale": list(self.scale), "intercept": self.intercept, "slopes": list(self.slopes)}

    @classmethod
    def from_json(cls, d: dict) -> "DiscriminationModel":
        return cls(d["name"], tuple(d["feature_names"]), tuple(d["mean"]), tuple(d["scale"]),
                   d["intercept"], tuple(d["slopes"]))


def _nll(params: np.ndarray, Z: np.ndarray, k: np.ndarray) -> float:
    p = np.clip(p_correct_from_z(params[0] + (Z * params[1:]).sum(1)), _EPS, 1 - _EPS)
    return float(-(k * np.log(p) + (N_SUBJECTS - k) * np.log(1 - p)).sum() + RIDGE * (params[1:] ** 2).sum())


def fit_model(name: str, X: np.ndarray, k: np.ndarray) -> DiscriminationModel:
    """X columns are the model's own features (in FEATURE_SETS[name] order)."""
    mean, scale = X.mean(0), X.std(0)
    if (scale < 1e-9).any():
        raise ValueError(f"{name}: a feature is constant in the training data; fit is undefined")
    Z = (X - mean) / scale
    best = None
    for a0 in (-1.0, 0.0, 1.0):  # a few starts: the sigmoid is flat in places
        res = minimize(_nll, np.r_[a0, np.zeros(Z.shape[1])], args=(Z, k), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    if not best.success:
        raise RuntimeError(f"fit of {name} did not converge: {best.message}")
    return DiscriminationModel(name, FEATURE_SETS[name], tuple(mean), tuple(scale),
                               float(best.x[0]), tuple(float(s) for s in best.x[1:]))


def fit_constant(k: np.ndarray) -> float:
    """MLE of a constant p_correct, constrained to >= chance."""
    return float(np.clip(k.sum() / (N_SUBJECTS * len(k)), CHANCE + 1e-6, 1 - 1e-6))


def per_triangle_logloss(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Binomial negative log-likelihood per trial (excludes the constant binomial coefficient)."""
    p = np.clip(p, _EPS, 1 - _EPS)
    return -(k * np.log(p) + (N_SUBJECTS - k) * np.log(1 - p)) / N_SUBJECTS


def _cols(ft: FeatureTable, name: str) -> np.ndarray:
    return ft.X[:, [FEATURES.index(f) for f in FEATURE_SETS[name]]]


# ------------------------------------------------------------------ evaluation
@dataclass
class OOF:
    """Out-of-fold predictions per seed: preds[model] has shape (n_seeds, n_triangles), NaN if not tested."""
    preds: dict[str, np.ndarray]
    const: np.ndarray


def _empty_oof(n_seeds: int, n: int) -> OOF:
    return OOF({m: np.full((n_seeds, n), np.nan) for m in FEATURE_SETS}, np.full((n_seeds, n), np.nan))


def _fit_predict(ft: FeatureTable, train: np.ndarray, test: np.ndarray, oof: OOF, seed_i: int) -> None:
    for m in FEATURE_SETS:
        X = _cols(ft, m)
        model = fit_model(m, X[train], ft.k[train])
        oof.preds[m][seed_i, test] = model.p_correct(X[test])
    oof.const[seed_i, test] = fit_constant(ft.k[train])


def grouped_cv(ft: FeatureTable, seeds=(0, 1, 2), folds=10) -> OOF:
    """Whole triangles held out (each triangle is one group: its 26 responses stay together)."""
    n = len(ft.triangles)
    oof = _empty_oof(len(seeds), n)
    for si, seed in enumerate(seeds):
        order = np.random.default_rng(seed).permutation(n)
        for f in range(folds):
            test = np.zeros(n, bool)
            test[order[f::folds]] = True
            _fit_predict(ft, ~test, test, oof, si)
    return oof


def molecule_holdout(ft: FeatureTable, all_cids: Sequence[int], seeds=(0, 1, 2), folds=5
                     ) -> tuple[OOF, dict]:
    """Hold out molecules: 5 folds over the 128 molecules; test = triangles whose differing
    components are >50% from the held-out fold; train = every other triangle.

    Why not also drop training triangles that merely contain held-out molecules: the model has only
    2-5 global parameters (nothing per molecule), so identity cannot leak, and the strict version
    (train on triangles with NO held-out differing molecule) leaves ~40-55 training triangles that
    are mostly one-differing-component triangles (overlap barely varies), so the fit cannot learn the dose-response. The strict
    training composition is recorded in the output for transparency."""
    n = len(ft.triangles)
    oof = _empty_oof(len(seeds), n)
    cids = sorted(all_cids)
    strict_train: list[dict] = []
    for si, seed in enumerate(seeds):
        perm = np.random.default_rng(seed).permutation(len(cids))
        fold_of = {cids[i]: j % folds for j, i in enumerate(perm)}
        for f in range(folds):
            frac = np.array([np.mean([fold_of[c] == f for c in t.differing_cids]) for t in ft.triangles])
            hit = np.array([any(fold_of[c] == f for c in t.differing_cids) for t in ft.triangles])
            test = frac > 0.5
            strict_train.append({"n_train": int((~hit).sum()),
                                 "frac_one_differing_component": round(float(np.mean([t.n_differ == 1 for t, h in zip(ft.triangles, hit) if not h])), 2)})
            if test.sum():
                _fit_predict(ft, ~test, test, oof, si)
    tested = ~np.isnan(oof.const).all(0)
    by_diff = {int(d): int(c) for d, c in zip(*np.unique([t.n_differ for t, x in zip(ft.triangles, tested) if x],
                                                         return_counts=True))}
    info = {"n_test_triangles_ever": int(tested.sum()),
            "n_test_per_seed": [int((~np.isnan(oof.const[s])).sum()) for s in range(len(seeds))],
            "test_triangles_by_n_differ": by_diff,
            "strict_split_diagnostic": {
                "n_train_range": [min(d["n_train"] for d in strict_train), max(d["n_train"] for d in strict_train)],
                "min_fraction_one_differing_component": min(d["frac_one_differing_component"] for d in strict_train),
                "reading": "strict split (no held-out molecule among training differing components) is degenerate; not scored"}}
    return oof, info


def _seed_mean(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns = never tested; filtered by caller
        return np.nanmean(a, axis=0)


def summarize(ft: FeatureTable, oof: OOF, n_boot=4000, rng_seed=0) -> dict:
    """Metrics on triangles that were tested at least once; predictions averaged over seeds."""
    tested = ~np.isnan(oof.const).all(0)
    k = ft.k[tested]
    frac = k / N_SUBJECTS
    # per-seed log-loss, then mean over seeds (not log-loss of an averaged prediction)
    def ll(p3: np.ndarray) -> np.ndarray:
        per = np.array([per_triangle_logloss(p3[s][tested], k) for s in range(p3.shape[0])])
        return _seed_mean(per)
    ll_m = {m: ll(oof.preds[m]) for m in FEATURE_SETS}
    ll_c = ll(oof.const)
    rng = np.random.default_rng(rng_seed)
    boots = rng.integers(0, len(k), size=(n_boot, len(k)))
    pm = {m: _seed_mean(oof.preds[m])[tested] for m in FEATURE_SETS}

    def ci(x: np.ndarray) -> list[float]:
        return [round(float(v), 5) for v in np.percentile(x, [2.5, 97.5])]

    out: dict = {"n_triangles_tested": int(tested.sum()),
                 "logloss_constant_rate": round(float(ll_c.mean()), 5), "models": {}}
    for m in FEATURE_SETS:
        sp = float(spearmanr(pm[m], frac).statistic)
        entry = {
            "logloss": round(float(ll_m[m].mean()), 5),
            "skill_vs_constant": round(1 - float(ll_m[m].mean()) / float(ll_c.mean()), 4),
            "gain_vs_constant": round(float((ll_c - ll_m[m]).mean()), 5),
            "spearman": round(sp, 4),
            "calibration_5bins": calibration(pm[m], frac),
        }
        if m != BASELINE:
            d = ll_m[BASELINE] - ll_m[m]  # positive = model better than baseline
            dsp = np.array([spearmanr(pm[m][b], frac[b]).statistic - spearmanr(pm[BASELINE][b], frac[b]).statistic
                            for b in boots[:1000]])
            lo, hi = np.percentile(d[boots].mean(1), [2.5, 97.5])
            entry["gain_vs_baseline_logloss"] = {
                "mean": round(float(d.mean()), 5), "ci95": ci(d[boots].mean(1)),
                "verdict": ("model better" if lo > 0 else "baseline better" if hi < 0 else "indistinguishable")}
            entry["spearman_minus_baseline"] = {
                "mean": round(float(spearmanr(pm[m], frac).statistic - spearmanr(pm[BASELINE], frac).statistic), 4),
                "ci95": ci(dsp)}
        out["models"][m] = entry
    return out


def calibration(pred: np.ndarray, frac: np.ndarray, bins=5) -> list[dict]:
    order = np.argsort(pred)
    out = []
    for idx in np.array_split(order, bins):
        out.append({"n": int(len(idx)), "predicted": round(float(pred[idx].mean()), 3),
                    "observed": round(float(frac[idx].mean()), 3)})
    return out


# ------------------------------------------------------------------ consistency with Snitz / Ravia
def descriptor_vs_similarity() -> dict:
    """Spearman of descriptor and structural angle distances against human similarity ratings
    (Snitz 2013: 360 pairs; Ravia 2020: 195 pairs). Negative = distance tracks dissimilarity."""
    import re
    import pyrfume
    out: dict = {}

    def cid_smiles(name: str) -> dict[int, str]:
        mol = pyrfume.load_data(f"{name}/molecules.csv")
        return {int(c): r["IsomericSMILES"] for c, r in mol.iterrows()
                if isinstance(r["IsomericSMILES"], str) and Chem.MolFromSmiles(r["IsomericSMILES"])}

    def run(name: str, pairs: list[tuple[list[int], list[int], float]], c2s: dict[int, str]) -> dict:
        space = structural_space(c2s.values())
        table = descriptor_table()
        covered = {c for c, s in c2s.items() if table.get(s) is not None}
        dp, ds, sim, n_skip = [], [], [], 0
        for a, b, y in pairs:
            sa, sb = [c2s[c] for c in a if c in c2s], [c2s[c] for c in b if c in c2s]
            if not sa or not sb:
                n_skip += 1
                continue
            d, _ = perceptual_distance(sa, sb)
            if d is None:
                n_skip += 1
                continue
            dp.append(d); ds.append(structural_distance(space, sa, sb)); sim.append(y)
        return {"n_pairs_used": len(sim), "n_pairs_skipped_no_descriptors": n_skip,
                "leffingwell_coverage_of_molecules": f"{len(covered)}/{len(c2s)}",
                "spearman_descriptor_distance_vs_similarity": round(float(spearmanr(dp, sim).statistic), 3),
                "spearman_structural_distance_vs_similarity": round(float(spearmanr(ds, sim).statistic), 3)}

    c2s = cid_smiles("snitz_2013")
    beh = pyrfume.load_data("snitz_2013/behavior.csv")
    toks = lambda cell: [int(t) for t in str(cell).split(",") if t.strip().isdigit()]  # noqa: E731
    out["snitz_2013"] = run("snitz_2013", [(toks(r["StimulusA"]), toks(r["StimulusB"]), float(r["Similarity"]))
                                            for _, r in beh.iterrows()], c2s)
    c2s = cid_smiles("ravia_2020")
    st = pyrfume.load_data("ravia_2020/stimuli.csv")
    b2 = pyrfume.load_data("ravia_2020/behavior_2.csv").reset_index()
    stim = lambda sid: ([int(t) for t in re.split(r"[;,]", str(st.loc[sid, "CID"])) if t.strip().isdigit()]  # noqa: E731
                        if sid in st.index else [])
    out["ravia_2020"] = run("ravia_2020", [(stim(r["Stimulus 1"]), stim(r["Stimulus 2"]), float(r["RatedSimilarity"]))
                                           for _, r in b2.iterrows()], c2s)
    return out


# ------------------------------------------------------------------ public API
@dataclass(frozen=True)
class Prediction:
    p_correct: float
    features: dict[str, float]
    missing_descriptor_smiles: list[str]


def _unique_canonical(smiles: Sequence[str], weights: Sequence[float] | None) -> set[str]:
    return {canonical(s) for s in smiles}


def predict_details(model: DiscriminationModel, mixture_a_smiles: Sequence[str], mixture_b_smiles: Sequence[str],
                    weights_a: Sequence[float] | None = None, weights_b: Sequence[float] | None = None,
                    allow_missing_descriptors: bool = False) -> Prediction:
    """p_correct for a triangle test of mixture A against B.

    Weights (optional, one per component) scale each component's contribution to the structural
    centroid and to the summed Leffingwell vector. They do NOT enter the overlap features, which
    are set-based. EXTRAPOLATION: Bushdid's components were roughly intensity-matched, so no
    weighted mixture was ever tested; weighted predictions assume perceived contribution is
    proportional to the weight.
    """
    if not mixture_a_smiles or not mixture_b_smiles:
        raise ValueError("mixtures must be non-empty")
    ca, cb = _unique_canonical(mixture_a_smiles, weights_a), _unique_canonical(mixture_b_smiles, weights_b)
    if len(ca) != len(mixture_a_smiles) or len(cb) != len(mixture_b_smiles):
        raise ValueError("duplicate component within a mixture")
    n = max(len(ca), len(cb))
    ref = load_bushdid().cid_to_smiles.values()
    space = _space_cached(tuple(ref))
    pdist, missing = perceptual_distance(mixture_a_smiles, mixture_b_smiles, weights_a, weights_b)
    if missing and not allow_missing_descriptors:
        raise ValueError(f"no Leffingwell descriptors for {sorted(set(missing))}; pass "
                         "allow_missing_descriptors=True to skip them (changes the distance)")
    if pdist is None:
        raise ValueError("a mixture has no component with Leffingwell descriptors")
    feats = {"overlap": len(ca & cb) / n, "log_n": math.log(n),
             "structural": structural_distance(space, mixture_a_smiles, mixture_b_smiles, weights_a, weights_b),
             "perceptual": pdist}
    x = np.array([[feats[f] for f in model.feature_names]])
    return Prediction(float(model.p_correct(x)[0]), feats, sorted(set(missing)))


_SPACES: dict[tuple, MixtureSpace] = {}


def _space_cached(ref: tuple[str, ...]) -> MixtureSpace:
    if ref not in _SPACES:
        _SPACES[ref] = structural_space(ref)
    return _SPACES[ref]


def predict_p_correct(mixture_a_smiles: Sequence[str], mixture_b_smiles: Sequence[str],
                      weights_a: Sequence[float] | None = None, weights_b: Sequence[float] | None = None,
                      model: DiscriminationModel | None = None) -> float:
    """Predicted fraction of people who pick the odd mixture in a triangle test (chance 1/3).
    Uses the saved default model (results/discrimination_model.json). See predict_details."""
    model = model or load_default()
    return predict_details(model, mixture_a_smiles, mixture_b_smiles, weights_a, weights_b).p_correct


def load_default() -> DiscriminationModel:
    return DiscriminationModel.from_json(json.loads((RESULTS_DIR / "discrimination_model.json").read_text()))


def fit_default(model_name: str = DEFAULT_MODEL, save: bool = True) -> DiscriminationModel:
    ft = build_features(load_bushdid())
    model = fit_model(model_name, _cols(ft, model_name), ft.k)
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "discrimination_model.json").write_text(json.dumps(
            {**model.to_json(), "fitted_on": "bushdid_2014, all triangles with defined features",
             "n_triangles": len(ft.triangles)}, indent=2))
    return model


# ------------------------------------------------------------------ main
def run_all() -> dict:
    data = load_bushdid()
    ft = build_features(data)
    results: dict = {"data_audit": ft.audit}
    n_diff = {int(d): int(c) for d, c in zip(*np.unique([t.n_differ for t in ft.triangles], return_counts=True))}
    results["triangles_by_n_differ"] = n_diff
    results["mean_fraction_correct"] = round(float(ft.k.mean() / N_SUBJECTS), 4)

    full = {m: fit_model(m, _cols(ft, m), ft.k) for m in FEATURE_SETS}
    results["full_data_fit"] = {}
    for m, mod in full.items():
        p = mod.p_correct(_cols(ft, m))
        pearson_disp = float((((ft.k - N_SUBJECTS * p) ** 2) / (N_SUBJECTS * p * (1 - p))).sum() / (len(ft.k) - 1 - len(mod.slopes)))
        results["full_data_fit"][m] = {"features": list(mod.feature_names), "intercept": round(mod.intercept, 4),
                                       "slopes_standardised": [round(s, 4) for s in mod.slopes],
                                       "in_sample_logloss": round(float(per_triangle_logloss(p, ft.k).mean()), 5),
                                       "pearson_dispersion": round(pearson_disp, 3)}
    results["grouped_cv_10fold_3seeds"] = summarize(ft, grouped_cv(ft))
    oof, info = molecule_holdout(ft, list(data.cid_to_smiles))
    results["molecule_holdout_5fold_3seeds"] = {**info, **summarize(ft, oof)}
    results["descriptor_vs_human_similarity"] = descriptor_vs_similarity()
    results["default_model"] = DEFAULT_MODEL
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "discrimination.json").write_text(json.dumps(results, indent=2))
    fit_default()
    return results


if __name__ == "__main__":
    res = run_all()
    for part in ("grouped_cv_10fold_3seeds", "molecule_holdout_5fold_3seeds"):
        print(part, res[part]["n_triangles_tested"], "const", res[part]["logloss_constant_rate"])
        for m, e in res[part]["models"].items():
            print(f"  {m:13s} ll={e['logloss']} skill={e['skill_vs_constant']} rho={e['spearman']}",
                  e.get("gain_vs_baseline_logloss", ""))
    print(res["data_audit"]); print(res["descriptor_vs_human_similarity"])
