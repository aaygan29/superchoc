"""Pre-registered analysis: structure-only prediction of odor-mixture pleasantness (Ma et al. 2021).

Implements exactly methods/chocolate_anchor/PREREGISTRATION.md (registered at commit 61a9465).
Component model trained on Keller 2016 only (every Ma molecule removed), mixtures scored with an
equal-weight mean of two predicted component values, tested on Ma's 198 binary mixtures.

Run (from repo root):
    PYTHONPATH=methods OMP_NUM_THREADS=1 .venv/bin/python -m chocolate_anchor.mixture_hedonics

Reusable API for scoring new candidate mixtures with the same models:
    ens = build_component_ensemble(exclude_cids)      # deterministic (seed 0)
    predict_components(smiles_list)                   # per-family + ensemble + agreement
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "methods" / "real_flavor"))  # real_flavor modules are top-level imports
from real_data import build_keller_pleasantness  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS_DIR = _HERE / "results"
MA_MIXTURES_CSV = _ROOT / "data" / "hedonics" / "mixture_pleasantness_ma2021.csv"
MA_BEHAVIOR_CSV = _ROOT / "data" / "hedonics" / "raw" / "ma_2021" / "behavior.csv"

SEED = 0
FAMILIES = ("rf_ecfp", "rf_desc", "hgb_desc", "knn_tanimoto")
N_BOOT = 10_000
N_PERM = 1_000
BOOT_SEED = 20261113
PERM_SEED_BASE = 1_000_000
DOMAIN_TANIMOTO = 0.4
ALPHA = 0.05

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_DESC_FUNCS = Descriptors.descList


# --------------------------------------------------------------------------------------
# structures and features
# --------------------------------------------------------------------------------------
def parse_smiles(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None


def stripped_smiles(mol: Chem.Mol) -> str:
    """Canonical stereo-stripped SMILES (the leakage-matching key)."""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


@dataclass(frozen=True)
class Features:
    ecfp: np.ndarray  # (n, 2048) uint8 ECFP4 bits
    desc: np.ndarray  # (n, n_desc) float64, inf -> nan, NOT yet imputed


def featurize(mols: Sequence[Chem.Mol]) -> Features:
    ecfp = np.zeros((len(mols), 2048), dtype=np.uint8)
    desc = np.full((len(mols), len(_DESC_FUNCS)), np.nan)
    for i, m in enumerate(mols):
        ecfp[i] = _MORGAN.GetFingerprintAsNumPy(m)
        for j, (_, fn) in enumerate(_DESC_FUNCS):
            try:
                desc[i, j] = fn(m)
            except Exception:  # descriptor undefined for this molecule -> imputed later
                desc[i, j] = np.nan
    desc[~np.isfinite(desc)] = np.nan
    return Features(ecfp, desc)


def tanimoto_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


# --------------------------------------------------------------------------------------
# component ensemble
# --------------------------------------------------------------------------------------
def _knn_predict(train_ecfp: np.ndarray, y: np.ndarray, query_ecfp: np.ndarray, k: int = 10) -> np.ndarray:
    """k-NN regression, Tanimoto distance (1 - sim), distance-weighted exactly as sklearn does
    (an exact match, distance 0, takes all the weight)."""
    dist = 1.0 - tanimoto_matrix(query_ecfp, train_ecfp)
    idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
    d = np.take_along_axis(dist, idx, axis=1)
    out = np.empty(len(query_ecfp))
    for i in range(len(out)):
        zero = d[i] == 0
        w = zero.astype(float) if zero.any() else 1.0 / d[i]
        out[i] = np.sum(w * y[idx[i]]) / w.sum()
    return out


@dataclass(frozen=True)
class FittedFamilies:
    """The four trained component predictors plus what is needed to apply them."""

    train_ecfp: np.ndarray
    train_y: np.ndarray
    desc_median: np.ndarray
    rf_ecfp: RandomForestRegressor
    rf_desc: RandomForestRegressor
    hgb_desc: HistGradientBoostingRegressor

    def predict_raw(self, feats: Features) -> np.ndarray:
        """(n, 4) predictions on the Keller 0-100 scale, columns in FAMILIES order."""
        desc = np.where(np.isnan(feats.desc), self.desc_median, feats.desc)
        return np.column_stack(
            [
                self.rf_ecfp.predict(feats.ecfp),
                self.rf_desc.predict(desc),
                self.hgb_desc.predict(desc),
                _knn_predict(self.train_ecfp, self.train_y, feats.ecfp),
            ]
        )


def fit_families(train: Features, y: np.ndarray, rf_n_jobs: int = 1) -> FittedFamilies:
    med = np.nanmedian(train.desc, axis=0)
    if np.isnan(med).any():
        raise ValueError("descriptor column undefined for every training molecule")
    desc = np.where(np.isnan(train.desc), med, train.desc)
    rf_ecfp = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=rf_n_jobs).fit(train.ecfp, y)
    rf_desc = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=rf_n_jobs).fit(desc, y)
    hgb = HistGradientBoostingRegressor(random_state=SEED).fit(desc, y)
    return FittedFamilies(train.ecfp, y, med, rf_ecfp, rf_desc, hgb)


def to_rank_scale(sorted_train_preds: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Empirical-CDF (mid-rank) of `preds` against `sorted_train_preds`, in [0, 1]. The reference is one family's
    IN-SAMPLE predictions on its own training molecules (see `_ensemble_from_features`), not the Keller labels' ranks:
    RF/kNN in-sample predictions are wider than their out-of-sample ones, so the families' scales are monotone
    transforms of their raw outputs but are not equally calibrated, and the equal-weight mean weights them unevenly."""
    n = len(sorted_train_preds)
    lo = np.searchsorted(sorted_train_preds, preds, side="left")
    hi = np.searchsorted(sorted_train_preds, preds, side="right")
    return (lo + hi) / (2.0 * n)


@dataclass(frozen=True)
class ComponentEnsemble:
    families: FittedFamilies
    sorted_train_preds: tuple[np.ndarray, ...]  # per family, sorted IN-SAMPLE predictions on the training molecules (rank-scale reference)
    keller_median: float  # median of Keller training labels (0-100 scale)
    train_cids: np.ndarray
    n_train: int

    def predict(self, mols: Sequence[Chem.Mol]) -> "ComponentPrediction":
        return self.predict_features(featurize(mols))

    def predict_features(self, feats: Features) -> "ComponentPrediction":
        raw = self.families.predict_raw(feats)
        rank = np.column_stack([to_rank_scale(self.sorted_train_preds[j], raw[:, j]) for j in range(4)])
        side = raw > self.keller_median
        return ComponentPrediction(
            raw=raw,
            family_rank=rank,
            ensemble=rank.mean(axis=1),
            agreement=side.all(axis=1) | (~side).all(axis=1),
        )


@dataclass(frozen=True)
class ComponentPrediction:
    raw: np.ndarray  # (n, 4) Keller-scale predictions
    family_rank: np.ndarray  # (n, 4) ECDF of each family's in-sample training predictions (NOT Keller label ranks), FAMILIES order
    ensemble: np.ndarray  # (n,) mean of the four rank-scale values
    agreement: np.ndarray  # (n,) bool: all four families on the same side of the Keller median


def _ensemble_from_features(train: Features, y: np.ndarray, cids: np.ndarray, rf_n_jobs: int) -> ComponentEnsemble:
    fam = fit_families(train, y, rf_n_jobs)
    in_sample = fam.predict_raw(train)
    return ComponentEnsemble(
        fam,
        tuple(np.sort(in_sample[:, j]) for j in range(4)),
        float(np.median(y)),
        cids,
        len(y),
    )


@dataclass(frozen=True)
class KellerTraining:
    cids: np.ndarray
    y: np.ndarray
    features: Features
    n_removed_by_cid: int
    n_removed_by_smiles: int
    n_removed_total: int
    n_invalid_smiles: int


def load_keller_training(exclude_cids: Iterable[int], exclude_smiles: Iterable[str] = ()) -> KellerTraining:
    """Keller 1/1000 pleasantness minus every molecule matching `exclude_cids` or `exclude_smiles`
    (canonical stereo-stripped). Asserts zero residual overlap."""
    excl_cids = {int(c) for c in exclude_cids}
    excl_smiles = set(exclude_smiles)
    kel = build_keller_pleasantness()
    mols = [parse_smiles(s) for s in kel.smiles]
    valid = np.array([m is not None for m in mols])
    kel, mols = kel[valid].reset_index(drop=True), [m for m in mols if m is not None]
    key = [stripped_smiles(m) for m in mols]
    cid = kel.CID.astype(int).to_numpy()
    by_cid = np.array([c in excl_cids for c in cid])
    by_smi = np.array([k in excl_smiles for k in key])
    drop = by_cid | by_smi
    keep = ~drop
    kept_mols = [m for m, k in zip(mols, keep) if k]
    assert not (set(cid[keep]) & excl_cids), "leakage: CID overlap"
    assert not ({k for k, kp in zip(key, keep) if kp} & excl_smiles), "leakage: SMILES overlap"
    return KellerTraining(
        cids=cid[keep],
        y=kel.pleasantness.to_numpy()[keep],
        features=featurize(kept_mols),
        n_removed_by_cid=int(by_cid.sum()),
        n_removed_by_smiles=int(by_smi.sum()),
        n_removed_total=int(drop.sum()),
        n_invalid_smiles=int((~valid).sum()),
    )


def build_component_ensemble(exclude_cids: Iterable[int], exclude_smiles: Iterable[str] = ()) -> ComponentEnsemble:
    """Train the four-family component ensemble on Keller minus `exclude_cids` (deterministic, seed 0)."""
    tr = load_keller_training(exclude_cids, exclude_smiles)
    return _ensemble_from_features(tr.features, tr.y, tr.cids, rf_n_jobs=-1)


_DEFAULT: ComponentEnsemble | None = None


def predict_components(smiles_list: Sequence[str], ensemble: ComponentEnsemble | None = None) -> dict:
    """Per-family (rank scale), ensemble and agreement for each SMILES.

    Default ensemble: the one validated here (Keller minus all Ma 2021 molecules).
    """
    global _DEFAULT
    if ensemble is None:
        if _DEFAULT is None:
            ma = load_ma_molecules()
            _DEFAULT = build_component_ensemble(ma.cid, ma.stripped)
        ensemble = _DEFAULT
    mols = [parse_smiles(s) for s in smiles_list]
    bad = [s for s, m in zip(smiles_list, mols) if m is None]
    if bad:
        raise ValueError(f"invalid SMILES: {bad}")
    p = ensemble.predict(mols)
    return {
        "families": {f: p.family_rank[:, j] for j, f in enumerate(FAMILIES)},
        "ensemble": p.ensemble,
        "agreement": p.agreement,
        "raw_keller_scale": {f: p.raw[:, j] for j, f in enumerate(FAMILIES)},
    }


# --------------------------------------------------------------------------------------
# Ma 2021 data
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MaMolecules:
    cid: np.ndarray
    name: list[str]
    smiles: list[str]
    mols: list[Chem.Mol]
    stripped: list[str]
    mw: np.ndarray
    heavy_atoms: np.ndarray
    excluded: list[dict]


def load_ma_molecules() -> MaMolecules:
    import pyrfume

    m = pyrfume.load_data("ma_2021/molecules.csv")
    cid, name, smi, mols, strp, mw, ha, excluded = [], [], [], [], [], [], [], []
    for c, row in m.iterrows():
        mol = parse_smiles(row.IsomericSMILES)
        if mol is None:
            excluded.append({"cid": int(c), "name": row["name"], "reason": "no valid structure"})
            continue
        cid.append(int(c)); name.append(str(row["name"])); smi.append(row.IsomericSMILES); mols.append(mol)
        strp.append(stripped_smiles(mol)); mw.append(float(row.MolecularWeight)); ha.append(mol.GetNumHeavyAtoms())
    return MaMolecules(np.array(cid), name, smi, mols, strp, np.array(mw), np.array(ha), excluded)


@dataclass(frozen=True)
class MaMixtures:
    table: pd.DataFrame  # one row per retained unique pair, with a_idx / b_idx into MaMolecules
    n_total: int
    excluded: list[dict]


def load_ma_mixtures(ma: MaMolecules) -> MaMixtures:
    df = pd.read_csv(MA_MIXTURES_CSV)
    beh = pd.read_csv(MA_BEHAVIOR_CSV)
    inmix = beh.groupby(["Stimulus A", "Stimulus B"])[["IAmix", "IBmix"]].mean().reset_index()
    ab = df.cids.str.split(";", expand=True).astype(int)
    df["cid_a"], df["cid_b"] = ab[0], ab[1]
    df = df.merge(inmix, left_on=["cid_a", "cid_b"], right_on=["Stimulus A", "Stimulus B"], how="left", validate="1:1")
    assert df.IAmix.notna().all() and df.IBmix.notna().all()
    pos = {c: i for i, c in enumerate(ma.cid)}
    keep = df.cid_a.isin(pos) & df.cid_b.isin(pos)
    excluded = [
        {"mixture_id": r.mixture_id, "reason": "component without valid structure"} for r in df[~keep].itertuples()
    ]
    df = df[keep].reset_index(drop=True)
    df["a_idx"] = df.cid_a.map(pos)
    df["b_idx"] = df.cid_b.map(pos)
    return MaMixtures(df, len(pd.read_csv(MA_MIXTURES_CSV)), excluded)


def component_alone_means(ma: MaMolecules, mix: MaMixtures) -> np.ndarray:
    """Per-molecule mean of measured alone-pleasantness over every mixture row it appears in."""
    t = mix.table
    vals: list[list[float]] = [[] for _ in ma.cid]
    for r in t.itertuples():
        vals[r.a_idx].append(r.pleas_A_alone)
        vals[r.b_idx].append(r.pleas_B_alone)
    return np.array([np.mean(v) if v else np.nan for v in vals])


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mixture_rule(comp_a: np.ndarray, comp_b: np.ndarray, w_a: np.ndarray | None = None,
                 w_b: np.ndarray | None = None) -> np.ndarray:
    """Predicted mixture value: equal-weight mean of the two component values (primary rule), or
    the weighted mean when weights are supplied (A4)."""
    if w_a is None or w_b is None:
        return (comp_a + comp_b) / 2.0
    return (w_a * comp_a + w_b * comp_b) / (w_a + w_b)


@dataclass(frozen=True)
class StatSpec:
    name: str
    kind: str  # "spearman_mix" | "auroc_mix" | "spearman_comp"
    score: str
    subset: str = "all"


@dataclass(frozen=True)
class BootData:
    """Arrays the bootstrap needs; resampling unit is the molecule."""

    n_mol: int
    a_idx: np.ndarray
    b_idx: np.ndarray
    obs: np.ndarray
    liked: np.ndarray
    mix_scores: dict[str, np.ndarray]
    subsets: dict[str, np.ndarray]
    comp_obs: np.ndarray
    comp_scores: dict[str, np.ndarray]


def eval_stat(d: BootData, s: StatSpec, drawn: np.ndarray, drawn_counts: np.ndarray) -> float:
    if s.kind == "spearman_comp":
        rows = np.repeat(np.arange(d.n_mol), drawn_counts)  # molecules with multiplicity
        return spearman(d.comp_scores[s.score][rows], d.comp_obs[rows])
    keep = d.subsets[s.subset] & drawn[d.a_idx] & drawn[d.b_idx]  # mixtures with both components drawn
    if s.kind == "spearman_mix":
        return spearman(d.mix_scores[s.score][keep], d.obs[keep])
    if s.kind == "auroc_mix":
        return auroc(d.mix_scores[s.score][keep], d.liked[keep])
    raise ValueError(s.kind)


def _boot_chunk(d: BootData, specs: Sequence[StatSpec], counts: np.ndarray) -> np.ndarray:
    out = np.empty((len(counts), len(specs)))
    for i, c in enumerate(counts):
        for j, s in enumerate(specs):
            out[i, j] = eval_stat(d, s, c > 0, c)
    return out


def cluster_bootstrap(d: BootData, specs: Sequence[StatSpec], n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                      n_jobs: int = -1) -> dict[str, np.ndarray]:
    """Molecule-cluster bootstrap: resample molecules with replacement; a mixture is kept iff both
    of its components were drawn. Resamples are generated up front so results do not depend on n_jobs."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, d.n_mol, size=(n_boot, d.n_mol))
    counts = np.apply_along_axis(lambda r: np.bincount(r, minlength=d.n_mol), 1, draws)
    chunks = np.array_split(counts, max(1, min(n_boot, 64)))
    parts = Parallel(n_jobs=n_jobs)(delayed(_boot_chunk)(d, specs, c) for c in chunks)
    arr = np.vstack(parts)
    return {s.name: arr[:, j] for j, s in enumerate(specs)}


def percentile_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, int]:
    ok = x[~np.isnan(x)]
    a = (1 - level) / 2
    return float(np.quantile(ok, a)), float(np.quantile(ok, 1 - a)), int(len(x) - len(ok))


def boot_p_le(x: np.ndarray, null: float = 0.0) -> float:
    """One-sided bootstrap p-value: share of resamples at or below `null` (with +1 correction)."""
    ok = x[~np.isnan(x)]
    return float((1 + np.sum(ok <= null)) / (len(ok) + 1))


def holm(p: dict[str, float]) -> dict[str, float]:
    order = sorted(p, key=p.get)
    m = len(order)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return adj


# --------------------------------------------------------------------------------------
# permutation test (shuffle Keller labels, retrain, re-predict)
# --------------------------------------------------------------------------------------
def _perm_one(i: int, train: Features, y: np.ndarray, ma_feats: Features, a_idx: np.ndarray, b_idx: np.ndarray,
              obs_mix: np.ndarray, obs_comp: np.ndarray) -> tuple[float, float]:
    perm_y = np.random.default_rng(PERM_SEED_BASE + i).permutation(y)
    ens = _ensemble_from_features(train, perm_y, np.empty(0), rf_n_jobs=1)
    comp = ens.predict_features(ma_feats).ensemble
    return spearman(comp, obs_comp), spearman(mixture_rule(comp[a_idx], comp[b_idx]), obs_mix)


def permutation_test(train: Features, y: np.ndarray, ma_feats: Features, a_idx: np.ndarray, b_idx: np.ndarray,
                     obs_mix: np.ndarray, obs_comp: np.ndarray, n_perm: int = N_PERM) -> np.ndarray:
    """(n_perm, 2) null Spearman of [component H1, mixture H2] under shuffled Keller labels."""
    res = Parallel(n_jobs=-1)(
        delayed(_perm_one)(i, train, y, ma_feats, a_idx, b_idx, obs_mix, obs_comp) for i in range(n_perm)
    )
    return np.array(res)


def perm_p(null: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


# --------------------------------------------------------------------------------------
# the analyses
# --------------------------------------------------------------------------------------
def _ci(arr: dict[str, np.ndarray], name: str, point: float) -> dict:
    lo, hi, n_nan = percentile_ci(arr[name])
    return {"estimate": point, "ci95": [lo, hi], "n_boot_undefined": n_nan}


def h1_components(arr, point, p_perm) -> dict:
    r = _ci(arr, "H1", point)
    r["permutation_p"] = p_perm
    r["criterion_met"] = bool(r["ci95"][0] > 0)
    return r


def h2_mixtures(arr, point, p_perm) -> dict:
    r = _ci(arr, "H2", point)
    r["permutation_p"] = p_perm
    r["criterion_met"] = bool(r["ci95"][0] > 0 and p_perm < ALPHA)
    return r


def h3_classification(arr, pt_conf, pt_abst, n_conf, n_abst) -> dict:
    conf, abst = _ci(arr, "H3_conf", pt_conf), _ci(arr, "H3_abst", pt_abst)
    diff = arr["H3_conf"] - arr["H3_abst"]
    ok = ~np.isnan(diff)
    dlo, dhi = np.quantile(diff[ok], [0.025, 0.975])
    p = max(boot_p_le(arr["H3_conf"], 0.5), boot_p_le(diff, 0.0))
    return {
        "auroc_confident": conf, "auroc_abstained": abst, "n_confident": n_conf, "n_abstained": n_abst,
        "difference_conf_minus_abst": {"estimate": None if np.isnan(pt_conf - pt_abst) else pt_conf - pt_abst,
                                       "ci95": [float(dlo), float(dhi)], "n_boot_valid": int(ok.sum())},
        "bootstrap_p": p,
        "criterion_met": bool(pt_conf > pt_abst and conf["ci95"][0] > 0.5),
    }


def code_provenance() -> dict:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=_ROOT, capture_output=True, text=True).stdout.strip()

    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_dirty_for_analysis_files": bool(
            git("status", "--porcelain", "--", "methods/chocolate_anchor/mixture_hedonics.py")
        ),
        "mixture_hedonics_py_sha256": hashlib.sha256((_HERE / "mixture_hedonics.py").read_bytes()).hexdigest(),
        "preregistration_commit": "61a9465",
    }


def run(n_boot: int = N_BOOT, n_perm: int = N_PERM) -> dict:
    t0 = time.time()
    ma = load_ma_molecules()
    mix = load_ma_mixtures(ma)
    t = mix.table
    tr = load_keller_training(ma.cid, ma.stripped)
    assert not (set(tr.cids.tolist()) & set(ma.cid.tolist()))

    ens = _ensemble_from_features(tr.features, tr.y, tr.cids, rf_n_jobs=-1)
    ma_feats = featurize(ma.mols)
    comp = ens.predict(ma.mols)
    a, b = t.a_idx.to_numpy(), t.b_idx.to_numpy()
    obs = t.pleasantness_mix.to_numpy()
    comp_obs = component_alone_means(ma, mix)
    ok_comp = ~np.isnan(comp_obs)
    assert ok_comp.all()

    mix_scores: dict[str, np.ndarray] = {"ensemble": mixture_rule(comp.ensemble[a], comp.ensemble[b])}
    for j, f in enumerate(FAMILIES):
        mix_scores[f] = mixture_rule(comp.family_rank[a, j], comp.family_rank[b, j])
    mix_scores["baseline_mw"] = mixture_rule(ma.mw[a], ma.mw[b])
    mix_scores["baseline_heavy_atoms"] = mixture_rule(ma.heavy_atoms[a].astype(float), ma.heavy_atoms[b].astype(float))
    mix_scores["baseline_constant"] = np.zeros(len(t))
    mix_scores["oracle_measured"] = mixture_rule(t.pleas_A_alone.to_numpy(), t.pleas_B_alone.to_numpy())
    mix_scores["intensity_weighted"] = mixture_rule(comp.ensemble[a], comp.ensemble[b], t.IAmix.to_numpy(), t.IBmix.to_numpy())

    comp_scores = {"ensemble": comp.ensemble, **{f: comp.family_rank[:, j] for j, f in enumerate(FAMILIES)},
                   "baseline_mw": ma.mw, "baseline_heavy_atoms": ma.heavy_atoms.astype(float)}

    liked = obs > np.median(obs)
    confident = comp.agreement[a] & comp.agreement[b]
    sims = tanimoto_matrix(ma_feats.ecfp, tr.features.ecfp).max(axis=1)
    in_domain = (sims[a] >= DOMAIN_TANIMOTO) & (sims[b] >= DOMAIN_TANIMOTO)
    subsets = {"all": np.ones(len(t), bool), "confident": confident, "abstained": ~confident,
               "domain_in": in_domain, "domain_out": ~in_domain}

    specs = [StatSpec("H1", "spearman_comp", "ensemble"), StatSpec("H2", "spearman_mix", "ensemble"),
             StatSpec("H3_conf", "auroc_mix", "ensemble", "confident"),
             StatSpec("H3_abst", "auroc_mix", "ensemble", "abstained"),
             StatSpec("H3_all", "auroc_mix", "ensemble")]
    for s in [*FAMILIES, "baseline_mw", "baseline_heavy_atoms", "oracle_measured", "intensity_weighted"]:
        specs.append(StatSpec(f"mix_spearman:{s}", "spearman_mix", s))
    for f in FAMILIES:
        specs.append(StatSpec(f"comp_spearman:{f}", "spearman_comp", f))
    for s in ("baseline_mw", "baseline_heavy_atoms"):
        specs.append(StatSpec(f"comp_spearman:{s}", "spearman_comp", s))
    specs += [StatSpec("A5_in", "spearman_mix", "ensemble", "domain_in"),
              StatSpec("A5_out", "spearman_mix", "ensemble", "domain_out")]
    data = BootData(len(ma.cid), a, b, obs, liked, mix_scores, subsets, comp_obs, comp_scores)

    full_drawn, full_counts = np.ones(len(ma.cid), bool), np.ones(len(ma.cid), int)
    point = {s.name: eval_stat(data, s, full_drawn, full_counts) for s in specs}
    boot = cluster_bootstrap(data, specs, n_boot)
    null = permutation_test(tr.features, tr.y, ma_feats, a, b, obs, comp_obs, n_perm)

    p_h1, p_h2 = perm_p(null[:, 0], point["H1"]), perm_p(null[:, 1], point["H2"])
    h3 = h3_classification(boot, point["H3_conf"], point["H3_abst"], int(confident.sum()), int((~confident).sum()))
    p_adj = holm({"H1": p_h1, "H2": p_h2, "H3": h3["bootstrap_p"]})
    h1, h2 = h1_components(boot, point["H1"], p_h1), h2_mixtures(boot, point["H2"], p_h2)
    for k, h in (("H1", h1), ("H2", h2), ("H3", h3)):
        raw = {"H1": p_h1, "H2": p_h2, "H3": h3["bootstrap_p"]}[k]
        h["p_raw"], h["p_holm"] = raw, p_adj[k]
        h["supported"] = bool(h["criterion_met"] and p_adj[k] < ALPHA)

    def row(name: str) -> dict:
        return {"mixture_spearman": _ci(boot, f"mix_spearman:{name}", point[f"mix_spearman:{name}"])}

    a1 = {f: {**row(f), "component_spearman": _ci(boot, f"comp_spearman:{f}", point[f"comp_spearman:{f}"])}
          for f in FAMILIES}
    a1["ensemble"] = {"mixture_spearman": h2_ci(boot, point), "component_spearman": _ci(boot, "H1", point["H1"])}

    def vs_ens(name: str) -> dict:
        d = boot["H2"] - boot[f"mix_spearman:{name}"]
        lo, hi, _ = percentile_ci(d)
        return {**row(name), "difference_ensemble_minus_this": {"estimate": point["H2"] - point[f"mix_spearman:{name}"],
                                                                 "ci95": [lo, hi]}}

    a2 = {"baseline_mw": {**vs_ens("baseline_mw"),
                          "component_spearman": _ci(boot, "comp_spearman:baseline_mw", point["comp_spearman:baseline_mw"])},
          "baseline_heavy_atoms": {**vs_ens("baseline_heavy_atoms"),
                                   "component_spearman": _ci(boot, "comp_spearman:baseline_heavy_atoms",
                                                             point["comp_spearman:baseline_heavy_atoms"])},
          "baseline_constant": {"mixture_spearman": None, "auroc": 0.5,
                                "note": "constant predictor: rank correlation undefined, AUROC 0.5 by construction"}}
    oracle_pearson = float(np.corrcoef(mix_scores["oracle_measured"], obs)[0, 1])
    a3 = {**vs_ens("oracle_measured"), "pearson": oracle_pearson}
    a4 = vs_ens("intensity_weighted")
    a5 = {"n_in_domain": int(in_domain.sum()), "n_out_of_domain": int((~in_domain).sum()),
          "in_domain": _ci(boot, "A5_in", point["A5_in"]), "out_of_domain": _ci(boot, "A5_out", point["A5_out"]),
          "n_molecules_max_tanimoto_ge_0.4": int((sims >= DOMAIN_TANIMOTO).sum()),
          "max_tanimoto_median": float(np.median(sims))}
    a6 = {"keller_scale": "0-100 (untrained raters), training labels mean %.1f sd %.1f range [%.1f, %.1f]"
                          % (tr.y.mean(), tr.y.std(), tr.y.min(), tr.y.max()),
          "ma_scale": "0-10 (trained panel), mixture panel means mean %.2f sd %.2f range [%.2f, %.2f]"
                      % (obs.mean(), obs.std(), obs.min(), obs.max()),
          "note": "only rank statistics (Spearman, AUROC) are used for claims"}

    # predictions table
    pred = pd.DataFrame({"mixture_id": t.mixture_id, "cids": t.cids, "components": t.components,
                         "observed": obs, "liked_above_median": liked,
                         **{f"pred_{f}": mix_scores[f] for f in FAMILIES}, "pred_ensemble": mix_scores["ensemble"],
                         "comp_a_agree": comp.agreement[a], "comp_b_agree": comp.agreement[b],
                         "confident": confident})
    RESULTS_DIR.mkdir(exist_ok=True)
    pred.to_csv(RESULTS_DIR / "mixture_hedonics_predictions.csv", index=False)

    result = {
        "n": {"ma_molecules_total": len(ma.cid) + len(ma.excluded), "ma_molecules_used": len(ma.cid),
              "ma_mixtures_total_unique_pairs": mix.n_total, "ma_mixtures_used": len(t),
              "keller_1_1000_molecules": int(tr.n_removed_total + len(tr.y) + tr.n_invalid_smiles),
              "keller_training_after_exclusion": int(len(tr.y)), "median_of_mixture_ratings": float(np.median(obs)),
              "n_liked": int(liked.sum())},
        "exclusions": {"ma_molecules_no_valid_structure": ma.excluded, "ma_mixtures_excluded": mix.excluded,
                       "keller_removed_for_leakage": {"by_cid": tr.n_removed_by_cid, "by_stripped_smiles": tr.n_removed_by_smiles,
                                                      "total_union": tr.n_removed_total},
                       "keller_invalid_smiles": tr.n_invalid_smiles,
                       "leakage_assertion": "zero Keller-training molecules share a CID or canonical stereo-stripped SMILES with any Ma molecule"},
        "H1": h1, "H2": h2, "H3": h3,
        "H3_auroc_all_mixtures": _ci(boot, "H3_all", point["H3_all"]),
        "A1_per_family": a1, "A2_baselines": a2, "A3_oracle_measured_components": a3,
        "A4_intensity_weighted_predicted_components": a4, "A5_applicability_domain": a5, "A6_scale_check": a6,
        "settings": {"seed_models": SEED, "bootstrap_n": n_boot, "bootstrap_seed": BOOT_SEED, "permutations": n_perm,
                     "permutation_seeds": f"default_rng({PERM_SEED_BASE} + i) for permutation i", "alpha": ALPHA,
                     "domain_tanimoto": DOMAIN_TANIMOTO, "keller_median_labels": ens.keller_median},
        "deviations_interpretations": DEVIATIONS,
        "code": code_provenance(),
        "runtime_seconds": time.time() - t0,
    }
    (RESULTS_DIR / "mixture_hedonics.json").write_text(json.dumps(_jsonable(result), indent=2))
    return result


def h2_ci(boot, point) -> dict:
    return _ci(boot, "H2", point["H2"])


DEVIATIONS = [
    "H1/H2 p-values for Holm: both come from the same label-permutation test (one-sided, (1+#null>=obs)/(1+n_perm)); the plan specifies the permutation test only for H2, and gives no p-value for H1 or H3.",
    "H3 p-value for Holm: bootstrap one-sided, max of P(AUROC_confident<=0.5) and P(AUROC_confident-AUROC_abstained<=0) (intersection of the two stated conditions), (1+#)/(1+B). Verdict = pre-registered criterion AND Holm-adjusted p < 0.05.",
    "Bootstrap: percentile CIs. Molecules resampled with replacement; for mixtures a mixture is kept once iff both components were drawn (not weighted by multiplicity); H1 keeps drawn molecules with multiplicity. Resamples with an undefined statistic (e.g. one class only) are dropped and counted (n_boot_undefined).",
    "Rank normalisation: each family's value is the mid-rank empirical CDF of that family's in-sample Keller-training predictions, in [0,1].",
    "'Keller-median' for the abstention rule: median of the Keller training labels (0-100 scale); each family's raw prediction is compared with it. A mixture is confident iff all four families agree on the side for component A and for component B (the two components need not lie on the same side). Ties (prediction == median) count as below.",
    "'Liked' = strictly above the median of the 198 observed mixture ratings.",
    "Duplicate pairs: the pre-computed 198-row Ma table (panel means over raters and repetitions) is used as given; H1 per-molecule observed value = mean of the per-mixture panel-mean alone ratings over all mixtures containing the molecule (A and B slots).",
    "Component identity is by CID; the one component named with a CAS string (CID 61592, ethyl 3-(methylthio)propionate) is resolved via pyrfume molecules.csv by CID. Ma SMILES (isomeric) are used for prediction; stereo-stripped canonical SMILES only for leakage matching.",
    "RDKit descriptors: inf set to NaN, then NaN -> training-set column median (the plan says NaN only). HistGradientBoosting given random_state=0 (no effect with defaults). Keller 1/1000 table is built by real_data.build_keller_pleasantness; molecules whose Keller SMILES RDKit cannot parse would be dropped and counted.",
    "A4 weights: panel-mean perceived intensity of each component in the mixture (behavior.csv IAmix/IBmix, mean over all rater rows of a pair, zeros included as recorded), linear (not squared) weights.",
    "A2: molecular weight (Ma molecules.csv) and heavy-atom count used directly as the score (no sign flipping), mixture value = mean of the two components; ensemble-minus-baseline difference CIs are from the same bootstrap resamples. Constant baseline has undefined rank correlation (reported AUROC 0.5).",
    "A5: 'in domain' = both components have max ECFP4 Tanimoto to the (Ma-excluded) Keller training set >= 0.4; otherwise out-of-domain. Descriptive; no permutation test.",
    "A3 oracle uses the per-mixture measured alone-ratings of that row (PA, PB), not molecule-averaged values.",
    "The permutation test retrains all four families per permutation (seed 0 models, Keller labels shuffled with default_rng(1000000+i)); the rank-normalisation is refit on the permuted-model training predictions.",
    "Joblib parallelism is used for bootstrap and permutations; results do not depend on n_jobs (resamples drawn up front, per-permutation seeds).",
]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main() -> None:
    r = run()
    for k in ("H1", "H2", "H3"):
        print(k, json.dumps(_jsonable(r[k]), indent=1))
    print("runtime", r["runtime_seconds"])


if __name__ == "__main__":
    # re-import under the package name so joblib workers can unpickle our dataclasses
    from chocolate_anchor.mixture_hedonics import main as _main

    _main()
