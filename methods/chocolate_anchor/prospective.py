"""FROZEN prediction set (v2 re-freeze after review) for a prospective tasting study replicating Ma 2021's binary-mixture protocol.

PROSPECTIVE (post-registration; not part of PREREGISTRATION.md). Produces, in one deterministic run:
  data/candidates/scored_pairs.csv                    every cocoa-anchored pair of eligible pool molecules, scored
  data/candidates/FROZEN_PREDICTIONS.csv              40 selected pairs + 8 Ma anchor mixtures (anchors are a descriptive
                                                      scale check, NOT prospective hypotheses)
  data/candidates/FROZEN_COMPONENT_PREDICTIONS.csv    one row per molecule used in the 40 pairs (component-level hypothesis)
  data/candidates/FROZEN_PREDICTIONS.sha256           "<sha256>  <filename>" for each of the two frozen CSVs
  results/prospective.json                            counts, settings, choices, code sha256, revision note

Design (every choice below is recorded in results/prospective.json):
  1. Model: the registered four-family component ensemble (`_ensemble_from_features`, seed 0, equal-weight mean mixture
     rule), trained on Keller with every Ma 2021 molecule AND every pool.csv molecule (eligible or not) removed,
     matched by CID and stereo-stripped canonical SMILES. Zero overlap is asserted, so every prospective component is
     unseen. Rank scale: ECDF of each family's in-sample training predictions (not Keller label ranks).
  2. Pairs: all unordered pairs of ELIGIBLE pool molecules with at least one role == cocoa_key.
     `score_pairs` takes tuples of component indices of any length (pairs now; ternaries later), sharded via joblib.
  3. No conformal intervals and no liked/disliked calls: calibration on Ma is not exchangeable with the cocoa set.
  4. Selection (deterministic): 5 equal-count quintiles of the full scored set by ensemble prediction, 8 per
     quintile, round-robin over quintiles; within a quintile the next pick minimises the summed prior use of its two
     molecules, then maximises component dissimilarity (1 - ECFP4 Tanimoto), then pair_id. No molecule more than
     3 times across the 40. Plus 8 Ma anchors at Ma-observed pleasantness deciles 1, 2, 3, 4, 7, 8, 9, 10: the mixture
     whose observed value is nearest the median observed value of that decile (nearest decile centre), among mixtures
     with both components purchasable (in pool.csv, excluded only for being in Ma) when any exist, else among all.

Run (repo root):  OMP_NUM_THREADS=1 PYTHONPATH=methods .venv/bin/python -m chocolate_anchor.prospective
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from chocolate_anchor import mixture_hedonics as mh

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
POOL_CSV = _ROOT / "data" / "candidates" / "pool.csv"
CANDIDATES_DIR = _ROOT / "data" / "candidates"
RESULTS_DIR = _HERE / "results"  # alongside the other chocolate_anchor result files

LABEL = "PROSPECTIVE (post-registration; not part of PREREGISTRATION.md)"
SEED = 0
N_PER_QUINTILE = 8
N_QUINTILES = 5
MAX_USES_PER_MOLECULE = 3
N_ANCHOR_BINS = 10
ANCHOR_DECILES = (1, 2, 3, 4, 7, 8, 9, 10)
DECIMALS = 6
PURCHASABLE_EXCLUSION = "in Ma 2021 data set (test data)"  # excluded only because it is a test molecule


# ======================================================================================
# molecules
# ======================================================================================
@dataclass(frozen=True)
class PoolMolecules:
    table: pd.DataFrame  # all pool.csv rows with `stripped` key and parsed mol column index
    mols: list


def load_pool() -> PoolMolecules:
    df = pd.read_csv(POOL_CSV, keep_default_na=False)
    mols = [mh.parse_smiles(s) for s in df.smiles]
    bad = [n for n, m in zip(df.name, mols) if m is None]
    assert not bad, f"invalid pool SMILES: {bad}"
    df = df.assign(stripped=[mh.stripped_smiles(m) for m in mols])
    assert df.eligible.dtype == bool
    return PoolMolecules(df, mols)


# ======================================================================================
# component table and sharded pair scoring
# ======================================================================================
@dataclass(frozen=True)
class ComponentTable:
    """Per-molecule predictions (rank scale), indexed by position."""

    ensemble: np.ndarray  # (n,)
    family: np.ndarray  # (n, 4), mh.FAMILIES order


def _score_shard(comp: ComponentTable, shard: np.ndarray) -> np.ndarray:
    """(len(shard), 1 + 4): ensemble then per-family equal-weight mean over the components of each tuple."""
    out = np.empty((len(shard), 5))
    out[:, 0] = comp.ensemble[shard].mean(axis=1)
    for j in range(4):
        out[:, 1 + j] = comp.family[:, j][shard].mean(axis=1)
    return out


def score_pairs(comp: ComponentTable, pairs: Sequence[Sequence[int]], n_jobs: int, shard_size: int) -> np.ndarray:
    """Equal-weight mixture rule over tuples of component indices (any equal length), in shards via joblib.
    Returns (n_pairs, 5): ensemble, then the four families in mh.FAMILIES order. Order follows `pairs`."""
    idx = np.asarray(pairs, dtype=int)
    assert idx.ndim == 2 and shard_size > 0
    shards = [idx[i:i + shard_size] for i in range(0, len(idx), shard_size)]
    parts = Parallel(n_jobs=n_jobs)(delayed(_score_shard)(comp, s) for s in shards)
    return np.vstack(parts) if parts else np.empty((0, 5))


# ======================================================================================
# selection
# ======================================================================================
def quintile_of_rank(pred: np.ndarray, pair_ids: Sequence[str]) -> np.ndarray:
    """Equal-count quintiles 0..4 by ensemble prediction; ties broken by pair_id so the split is deterministic."""
    order = np.lexsort((np.asarray(pair_ids), pred))
    q = np.empty(len(pred), dtype=int)
    q[order] = (np.arange(len(pred)) * N_QUINTILES) // len(pred)
    return q


def select_pairs(quintile: np.ndarray, comp_idx: np.ndarray, dissimilarity: np.ndarray, pair_ids: Sequence[str],
                 per_quintile: int = N_PER_QUINTILE, max_uses: int = MAX_USES_PER_MOLECULE) -> list[int]:
    """Greedy round-robin over quintiles. `comp_idx` (n_pairs, k) holds molecule indices; `dissimilarity` is the
    per-pair mean component dissimilarity. Raises if a quintile cannot be filled under the use cap."""
    uses: dict[int, int] = {}
    chosen: list[int] = []
    remaining = {q: [i for i in np.flatnonzero(quintile == q)] for q in range(N_QUINTILES)}
    picks = {q: 0 for q in range(N_QUINTILES)}
    for _ in range(per_quintile):
        for q in range(N_QUINTILES):
            feasible = [i for i in remaining[q] if all(uses.get(int(m), 0) < max_uses for m in comp_idx[i])]
            if not feasible:
                raise RuntimeError(f"quintile {q + 1} cannot reach {per_quintile} pairs under the use cap")
            best = min(feasible, key=lambda i: (sum(uses.get(int(m), 0) for m in comp_idx[i]), -dissimilarity[i], pair_ids[i]))
            remaining[q].remove(best)
            chosen.append(best)
            picks[q] += 1
            for m in comp_idx[best]:
                uses[int(m)] = uses.get(int(m), 0) + 1
    return chosen


def select_anchors(obs: np.ndarray, purchasable: np.ndarray, mixture_ids: Sequence[str]) -> list[tuple[int, int, bool]]:
    """(row index, decile 1..10, both_purchasable) of the Ma mixture nearest the median observed value of each
    anchor decile; both-purchasable rows preferred when any exist in that decile."""
    order = np.lexsort((np.asarray(mixture_ids), obs))
    q = np.empty(len(obs), dtype=int)
    q[order] = (np.arange(len(obs)) * N_ANCHOR_BINS) // len(obs) + 1
    out = []
    for k in ANCHOR_DECILES:
        members = np.flatnonzero(q == k)
        centre = float(np.median(obs[members]))
        pool = members[purchasable[members]] if purchasable[members].any() else members
        best = min(pool, key=lambda i: (abs(obs[i] - centre), mixture_ids[i]))
        out.append((int(best), k, bool(purchasable[best])))
    return out


# ======================================================================================
# run
# ======================================================================================
def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=_ROOT, capture_output=True, text=True).stdout.strip()


def _round(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(DECIMALS)


def run(candidates_dir: Path = CANDIDATES_DIR, results_dir: Path = RESULTS_DIR, n_jobs: int = -1,
        shard_size: int = 64) -> dict:
    # ---- molecules and exclusion set
    ma = mh.load_ma_molecules()
    mix = mh.load_ma_mixtures(ma)
    pool = load_pool()
    pt = pool.table
    excl_cids = set(int(c) for c in ma.cid) | set(int(c) for c in pt.cid)
    excl_smiles = set(ma.stripped) | set(pt.stripped)

    tr = mh.load_keller_training(excl_cids, excl_smiles)
    assert not (set(tr.cids.tolist()) & excl_cids), "leakage: CID overlap with Ma/pool"
    # rf_n_jobs=1: multithreaded forest prediction sums trees in arbitrary order, and the resulting float noise flips
    # rank-scale ties against the in-sample training predictions; single-threaded is bit-reproducible
    ens = mh._ensemble_from_features(tr.features, tr.y, tr.cids, rf_n_jobs=1)
    n_ma_in_keller = mh.load_keller_training([], []).y.shape[0] - mh.load_keller_training(ma.cid, ma.stripped).y.shape[0]
    n_pool_extra = mh.load_keller_training(ma.cid, ma.stripped).y.shape[0] - len(tr.y)

    # ---- Ma component predictions (anchor rows only)
    ma_comp = ens.predict(ma.mols)
    t = mix.table
    a, b = t.a_idx.to_numpy(), t.b_idx.to_numpy()
    obs = t.pleasantness_mix.to_numpy()
    ma_pred = mh.mixture_rule(ma_comp.ensemble[a], ma_comp.ensemble[b])

    # ---- pool components and pairs
    elig = pt[pt.eligible].reset_index(drop=True)
    elig_mols = [pool.mols[i] for i in pt.index[pt.eligible]]
    feats = mh.featurize(elig_mols)
    cp = ens.predict_features(feats)
    comp = ComponentTable(cp.ensemble, cp.family_rank)
    sim_train = mh.tanimoto_matrix(feats.ecfp, tr.features.ecfp).max(axis=1)
    sim_pool = mh.tanimoto_matrix(feats.ecfp, feats.ecfp)
    is_key = (elig.role == "cocoa_key").to_numpy()
    n = len(elig)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if is_key[i] or is_key[j]]
    scores = score_pairs(comp, pairs, n_jobs=n_jobs, shard_size=shard_size)
    pidx = np.array(pairs)

    def pair_frame(ia: np.ndarray, ib: np.ndarray) -> pd.DataFrame:
        # order each pair by CID so names/cids/smiles columns are canonical
        swap = elig.cid.to_numpy()[ia] > elig.cid.to_numpy()[ib]
        ia, ib = np.where(swap, ib, ia), np.where(swap, ia, ib)
        col = lambda c, ix: elig[c].to_numpy()[ix]  # noqa: E731
        return pd.DataFrame({
            "pair_id": [f"pool_{x}_{y}" for x, y in zip(col("cid", ia), col("cid", ib))],
            "name_a": col("name", ia), "name_b": col("name", ib),
            "cid_a": col("cid", ia), "cid_b": col("cid", ib),
            "smiles_a": col("smiles", ia), "smiles_b": col("smiles", ib),
            "role_a": col("role", ia), "role_b": col("role", ib),
            "anchor": False, "ma_mixture_id": "", "anchor_decile": 0,
            "max_tanimoto_train_a": sim_train[ia], "max_tanimoto_train_b": sim_train[ib],
            "component_tanimoto": sim_pool[ia, ib],
        })

    sp = pair_frame(pidx[:, 0], pidx[:, 1])
    # the swap above may reorder a/b relative to `pairs`, but the score columns are symmetric, so they stay aligned
    for j, f in enumerate(mh.FAMILIES):
        sp[f"pred_{f}"] = scores[:, 1 + j]
    sp["pred_ensemble"] = scores[:, 0]
    quint = quintile_of_rank(sp.pred_ensemble.to_numpy(), sp.pair_id.tolist())
    sp["prediction_quintile"] = quint + 1
    assert sp.pair_id.is_unique

    # ---- selection of 40
    comp_idx = np.column_stack([sp.cid_a.to_numpy(), sp.cid_b.to_numpy()])
    chosen = select_pairs(quint, comp_idx, 1.0 - sp.component_tanimoto.to_numpy(), sp.pair_id.tolist())
    sel = sp.iloc[chosen].copy()
    sel["selection_order"] = np.arange(1, len(sel) + 1)

    # ---- anchors from Ma
    pool_by_cid = pt.set_index("cid")
    purchasable_cids = set(pt[(~pt.eligible) & (pt.exclusion_reason == PURCHASABLE_EXCLUSION)].cid)
    purch = np.array([(int(x) in purchasable_cids) and (int(y) in purchasable_cids) for x, y in zip(t.cid_a, t.cid_b)])
    anchors = select_anchors(obs, purch, t.mixture_id.tolist())
    a_rows = []
    for row, k, both in anchors:
        ia, ib = int(a[row]), int(b[row])
        cid_a, cid_b = int(ma.cid[ia]), int(ma.cid[ib])
        if cid_a > cid_b:
            ia, ib, cid_a, cid_b = ib, ia, cid_b, cid_a
        role = lambda c: pool_by_cid.role[c] if c in pool_by_cid.index else "ma_only"  # noqa: E731
        fam = lambda j: (ma_comp.family_rank[ia, j] + ma_comp.family_rank[ib, j]) / 2.0  # noqa: E731
        pe = float(ma_pred[row])
        rec = {
            "pair_id": f"anchor_{t.mixture_id[row]}", "name_a": ma.name[ia], "name_b": ma.name[ib],
            "cid_a": cid_a, "cid_b": cid_b, "smiles_a": ma.smiles[ia], "smiles_b": ma.smiles[ib],
            "role_a": role(cid_a), "role_b": role(cid_b), "anchor": True, "ma_mixture_id": t.mixture_id[row],
            "anchor_decile": k,
            "max_tanimoto_train_a": float(mh.tanimoto_matrix(mh.featurize([ma.mols[ia]]).ecfp, tr.features.ecfp).max()),
            "max_tanimoto_train_b": float(mh.tanimoto_matrix(mh.featurize([ma.mols[ib]]).ecfp, tr.features.ecfp).max()),
            "component_tanimoto": float(mh.tanimoto_matrix(mh.featurize([ma.mols[ia]]).ecfp, mh.featurize([ma.mols[ib]]).ecfp)[0, 0]),
            **{f"pred_{f}": float(fam(j)) for j, f in enumerate(mh.FAMILIES)},
            "pred_ensemble": pe,
            "prediction_quintile": 0, "selection_order": 0,
        }
        a_rows.append(rec)
    anc = pd.DataFrame(a_rows)
    frozen = pd.concat([sel, anc], ignore_index=True)

    # ---- write
    candidates_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    sp_path = candidates_dir / "scored_pairs.csv"
    fz_path = candidates_dir / "FROZEN_PREDICTIONS.csv"
    _round(sp.assign(selected=sp.pair_id.isin(sel.pair_id))).to_csv(sp_path, index=False)
    _round(frozen).to_csv(fz_path, index=False)
    # component-level hypothesis: one row per molecule used in the 40 test pairs
    used = sorted(set(sel.cid_a) | set(sel.cid_b))
    pos = {int(c): i for i, c in enumerate(elig.cid)}
    ix = np.array([pos[int(c)] for c in used])
    comp_df = pd.DataFrame({
        "cid": used, "name": elig.name.to_numpy()[ix], "smiles": elig.smiles.to_numpy()[ix],
        **{f"pred_{f}": comp.family[ix, j] for j, f in enumerate(mh.FAMILIES)},
        "pred_ensemble": comp.ensemble[ix], "max_tanimoto_train": sim_train[ix],
    })
    cp_path = candidates_dir / "FROZEN_COMPONENT_PREDICTIONS.csv"
    _round(comp_df).to_csv(cp_path, index=False)
    digest, comp_digest = _sha256_file(fz_path), _sha256_file(cp_path)
    (candidates_dir / "FROZEN_PREDICTIONS.sha256").write_text(
        f"{digest}  {fz_path.name}\n{comp_digest}  {cp_path.name}\n")

    uses = pd.concat([sel.cid_a, sel.cid_b]).value_counts()
    result = {
        "label": LABEL,
        "counts": {
            "eligible_pool_molecules": n, "eligible_cocoa_key": int(is_key.sum()), "eligible_partner": int((~is_key).sum()),
            "pairs_scored": len(sp), "frozen_rows": len(frozen), "prospective_pairs": len(sel), "anchors": len(anc),
            "molecules_in_selected_40": int(len(uses)), "max_uses_of_a_molecule": int(uses.max()),
            "selected_per_prediction_quintile": {int(k): int(v) for k, v in sel.prediction_quintile.value_counts().sort_index().items()},
            "anchors_both_purchasable": [bool(x[2]) for x in anchors],
        },
        "training": {
            "keller_1_1000_after_all_exclusions": int(len(tr.y)),
            "removed_by_cid": tr.n_removed_by_cid, "removed_by_stripped_smiles": tr.n_removed_by_smiles,
            "removed_total_union": tr.n_removed_total,
            "removed_for_ma_alone": n_ma_in_keller, "removed_additionally_for_pool": n_pool_extra,
            "ma_molecules": len(ma.cid), "pool_molecules_all_rows": len(pt),
            "leakage_assertion": "zero Keller-training molecules share a CID or canonical stereo-stripped SMILES with any Ma or pool.csv molecule",
        },
        "revision": "v2 re-freeze after review, nothing tasted: conformal intervals and three-way calls dropped, 8 anchors at "
                    "Ma deciles instead of 4 at quintiles, beta-damascenone made ineligible, component predictions frozen. "
                    "Old FROZEN_PREDICTIONS.csv hash 7c8eebdc41303cc12ce3a638de860a19e4ce91d2168cea1198d85644897139ab superseded.",
        "settings": {"seed_models": SEED, "families": list(mh.FAMILIES), "mixture_rule": "equal-weight mean of component rank-scale values",
                     "pair_rule": "unordered pairs of eligible pool molecules, at least one role == cocoa_key",
                     "score_pairs": {"n_jobs": n_jobs, "shard_size": shard_size},
                     "selection": {"quintiles": N_QUINTILES, "per_quintile": N_PER_QUINTILE, "max_uses_per_molecule": MAX_USES_PER_MOLECULE,
                                   "quintile_definition": "rank by ensemble prediction (ties by pair_id), position*5//n",
                                   "greedy": "round-robin over quintiles; each pick minimises summed prior uses of its two molecules, then maximises 1 - ECFP4 Tanimoto of the pair, then pair_id",
                                   "anchor_deciles_of_ma_observed": list(ANCHOR_DECILES),
                                   "anchor_rule": "mixture nearest the median observed value of the decile, both components purchasable preferred; descriptive scale check only",
                                   "purchasable": f"pool row with exclusion_reason == {PURCHASABLE_EXCLUSION!r}"},
                     "csv_float_decimals": DECIMALS},
        "frozen_predictions_sha256": digest,
        "frozen_component_predictions_sha256": comp_digest,
        "code": {"prospective_py_sha256": _sha256_file(_HERE / "prospective.py"),
                 "mixture_hedonics_py_sha256": _sha256_file(_HERE / "mixture_hedonics.py"),
                 "exploratory_py_sha256": _sha256_file(_HERE / "exploratory.py"),
                 "pool_csv_sha256": _sha256_file(POOL_CSV), "git_head": _git("rev-parse", "HEAD")},
        "interpretations": [
            "Anchors are scored by the same model as the pool pairs; the model never saw any Ma molecule, so anchor predictions are out-of-sample.",
            "Anchor 'eligible-for-purchase' = in pool.csv and excluded solely for being a Ma test molecule (myrcene, FDA-delisted, is therefore not purchasable).",
        ],
    }
    (results_dir / "prospective.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    r = run()
    print(json.dumps({k: r[k] for k in ("counts", "training", "frozen_predictions_sha256",
                                                        "frozen_component_predictions_sha256")}, indent=2))


if __name__ == "__main__":
    main()
