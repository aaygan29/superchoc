"""EXPLORATORY (post-registration) robustness checks requested after adversarial review.

  a. Fitted simple baselines (ridge and RF on 8 RDKit descriptors) vs the registered ensemble on the 198 Ma mixtures,
     paired molecule-cluster bootstrap of the Spearman difference.
  b. Near-duplicate sensitivity: H2 restricted to mixtures whose two components have max ECFP4 Tanimoto to the
     training set below 0.9 / 0.7.
  c. The prospective model (Ma AND pool molecules removed from training) on the Ma mixtures, and its agreement with
     the registered model.
  d. `cluster_bootstrap_lower_bound`: molecule-clustered inference for the prospective P1 design, with power and
     type-I error of the rule "permutation p < 0.05 AND cluster lower bound > 0" by Gaussian-copula simulation on
     the frozen 40 pairs' actual molecule structure.

Run (repo root):  OMP_NUM_THREADS=1 PYTHONPATH=methods .venv/bin/python -m chocolate_anchor.robustness
Outputs: results/robustness.json, ROBUSTNESS.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from chocolate_anchor import exploratory as ex
from chocolate_anchor import mixture_hedonics as mh
from chocolate_anchor import prospective as pr

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"
OUT_JSON = RESULTS_DIR / "robustness.json"
OUT_MD = _HERE / "ROBUSTNESS.md"
FROZEN_CSV = pr.CANDIDATES_DIR / "FROZEN_PREDICTIONS.csv"

LABEL = "EXPLORATORY (post-registration; not part of PREREGISTRATION.md)"
DESCRIPTORS = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount", "n_S_atoms")
RIDGE_ALPHA = 1.0
RF_TREES = 500
NEAR_DUPLICATE_CUTOFFS = (0.9, 0.7)
SIM_SEED = 20261117
N_SIMS = 2000
N_SIM_BOOT = 1000
N_SIM_PERM_NULL = 100_000
SIM_ALPHA = 0.05
SIM_RHOS = (0.49, 0.3, 0.0)
SIM_N_RATERS = 30


# ======================================================================================
# a. simple fitted baselines
# ======================================================================================
def simple_descriptors(mols: Sequence[Chem.Mol]) -> np.ndarray:
    """(n, 8): MolWt, MolLogP, TPSA, HBD, HBA, rotatable bonds, ring count, number of sulfur atoms."""
    return np.array([[Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
                      rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                      rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                      sum(a.GetSymbol() == "S" for a in m.GetAtoms())] for m in mols], dtype=float)


def keller_training_mols(tr: mh.KellerTraining) -> list[Chem.Mol]:
    """Molecules of `tr`, in order (KellerTraining stores features only)."""
    from real_data import build_keller_pleasantness

    kel = build_keller_pleasantness()
    by_cid = {int(c): s for c, s in zip(kel.CID, kel.smiles)}
    mols = [mh.parse_smiles(by_cid[int(c)]) for c in tr.cids]
    assert all(m is not None for m in mols)
    return mols


def fit_simple_baselines(train_x: np.ndarray, y: np.ndarray) -> dict:
    """Ridge (alpha 1, standardised inputs) and RF (500 trees, seed 0); each returns a predict(x) callable."""
    scaler = StandardScaler().fit(train_x)
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(train_x), y)
    rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=mh.SEED, n_jobs=1).fit(train_x, y)
    return {"ridge": lambda x: ridge.predict(scaler.transform(x)), "rf8": rf.predict}


# ======================================================================================
# d. molecule-clustered inference for P1
# ======================================================================================
def _index(comp_a: Sequence, comp_b: Sequence) -> tuple[np.ndarray, np.ndarray, int]:
    ids, inv = np.unique(np.concatenate([np.asarray(comp_a), np.asarray(comp_b)]), return_inverse=True)
    n = len(comp_a)
    return inv[:n], inv[n:], len(ids)


def _draw_counts(rng: np.random.Generator, n_boot: int, n_mol: int) -> np.ndarray:
    """(n_boot, n_mol) multiplicity of each molecule in each molecule-level resample."""
    draws = rng.integers(0, n_mol, size=(n_boot, n_mol))
    return (draws[:, :, None] == np.arange(n_mol)).sum(axis=1)


def _boot_rhos_loop(pred: np.ndarray, obs: np.ndarray, ia: np.ndarray, ib: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Tie-safe: Spearman of each resample over the pairs whose both components were drawn (NaN if < 3 kept)."""
    out = np.empty(len(counts))
    for i, c in enumerate(counts):
        keep = (c[ia] > 0) & (c[ib] > 0)
        out[i] = mh.spearman(pred[keep], obs[keep])
    return out


def _boot_rhos_vectorized(pred: np.ndarray, obs: np.ndarray, ia: np.ndarray, ib: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Same statistic, vectorised; exact only when pred and obs have no ties (true for simulated continuous data)."""
    keep = (counts[:, ia] > 0) & (counts[:, ib] > 0)  # (B, n_pairs)
    k = keep.sum(axis=1)

    def masked_rank(x: np.ndarray) -> np.ndarray:
        r = np.argsort(np.argsort(np.where(keep, x[None, :], np.inf), axis=1), axis=1).astype(float)
        return r - (k[:, None] - 1) / 2.0  # kept entries hold ranks 0..k-1; centre them

    rp, ro = masked_rank(pred), masked_rank(obs)
    num = (keep * rp * ro).sum(axis=1)
    den = np.sqrt((keep * rp ** 2).sum(axis=1) * (keep * ro ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = num / den
    rho[k < 3] = np.nan
    return rho


def cluster_bootstrap_lower_bound(pred: np.ndarray, obs: np.ndarray, comp_a: Sequence, comp_b: Sequence,
                                  level: float = 0.95, n_boot: int = mh.N_BOOT, seed: int = mh.BOOT_SEED) -> float:
    """One-sided percentile lower bound of the Spearman(pred, obs) across pairs, resampling MOLECULES with replacement
    (pairs share molecules, so pairs are not independent). A pair is kept in a resample iff both components were drawn.
    Resamples with fewer than 3 kept pairs are dropped. `comp_a`/`comp_b` are the molecule ids of each pair."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ia, ib, n_mol = _index(comp_a, comp_b)
    counts = _draw_counts(np.random.default_rng(seed), n_boot, n_mol)
    rho = _boot_rhos_loop(pred, obs, ia, ib, counts)
    return float(np.quantile(rho[~np.isnan(rho)], 1 - level))


def simulate_p1_rule(comp_a: Sequence, comp_b: Sequence, true_rho: float, vc: dict, latent_rho: float, crit: float,
                     n_sims: int, n_boot: int, seed: int) -> dict:
    """Power (or, at true_rho 0, type-I error) of the P1 rule on a design with the given molecule structure.

    Gaussian copula as in exploratory.simulate_cell, with the pair-level latents built from per-molecule latents
    (pair = (z_a + z_b)/sqrt 2) so pairs sharing a molecule are correlated: true latent T and predicted latent P of a
    molecule correlate `latent_rho`, hence so do pair-level T and P. T is mapped onto Ma's shrunken mean distribution by
    rank; panel mean = mu + mean of SIM_N_RATERS resampled Ma residuals. Rule: Spearman above the Monte-Carlo
    permutation critical value `crit` (= permutation p < 0.05) AND cluster lower bound (level 0.95) > 0."""
    rng = np.random.default_rng(seed)
    ia, ib, n_mol = _index(comp_a, comp_b)
    n = len(ia)
    r = 2 * math.sin(math.pi * latent_rho / 6)
    perm_hits = rule_hits = 0
    rhos = np.empty(n_sims)
    for s in range(n_sims):
        zt, e = rng.normal(size=n_mol), rng.normal(size=n_mol)
        zp = r * zt + math.sqrt(1 - r * r) * e
        t_lat, p_lat = (zt[ia] + zt[ib]) / math.sqrt(2), (zp[ia] + zp[ib]) / math.sqrt(2)
        mu_sorted = np.sort(rng.choice(vc["mu_hat"] - vc["mu_hat"].mean(), size=n) * vc["shrink"])
        mu = mu_sorted[np.argsort(np.argsort(t_lat))]
        panel = mu + rng.choice(vc["resid"], size=(n, SIM_N_RATERS)).mean(axis=1)
        rho = mh.spearman(p_lat, panel)
        rhos[s] = rho
        if rho > crit:
            perm_hits += 1
            counts = _draw_counts(rng, n_boot, n_mol)
            boot = _boot_rhos_vectorized(p_lat, panel, ia, ib, counts)
            lb = np.quantile(boot[~np.isnan(boot)], 1 - 0.95)
            rule_hits += int(lb > 0)
    return {"true_rho": true_rho, "latent_rho": latent_rho, "mean_observed_rho": float(rhos.mean()),
            "p_permutation_alone": perm_hits / n_sims, "p_permutation_and_cluster_lb": rule_hits / n_sims}


def p1_simulation(frozen: pd.DataFrame) -> dict:
    mixtures, _, _ = ex.load_rater_data()
    vc = ex.fit_variance_components(mixtures)
    rel = vc["sigma_mu"] ** 2 / (vc["sigma_mu"] ** 2 + vc["sigma_e"] ** 2 / SIM_N_RATERS)  # design-matched, as exploratory
    tests = frozen[~frozen.anchor]
    n = len(tests)
    crit = ex.spearman_critical(n, SIM_ALPHA, SIM_SEED)
    out = {"n_pairs": n, "n_molecules": int(len(set(tests.cid_a) | set(tests.cid_b))), "n_sims": N_SIMS,
           "n_boot_per_sim": N_SIM_BOOT, "design_matched_reliability_30_raters": rel, "spearman_critical_p05": crit,
           "scenarios": {}}
    for rho in SIM_RHOS:
        lat = min(rho / math.sqrt(rel), 0.99)
        out["scenarios"][f"{rho}"] = simulate_p1_rule(tests.cid_a.to_numpy(), tests.cid_b.to_numpy(), rho, vc, lat, crit,
                                                      N_SIMS, N_SIM_BOOT, SIM_SEED + int(rho * 100))
    out["assumptions"] = ("Gaussian copula; true rho is the expected Spearman with a 30-rater panel (disattenuated by design-matched "
                          "reliability); molecule-level latents induce the shared-molecule correlation; permutation p<0.05 is "
                          "approximated by the Monte-Carlo critical value for n=40; 1000 bootstrap resamples per simulated study.")
    return out


# ======================================================================================
# run
# ======================================================================================
def _ci(boot: dict, name: str, point: float) -> dict:
    lo, hi, n_nan = mh.percentile_ci(boot[name])
    return {"estimate": point, "ci95": [lo, hi], "n_nan_resamples": n_nan}


def run(n_boot: int = mh.N_BOOT, n_jobs: int = -1, simulate: bool = True, out_json: Path = OUT_JSON,
        out_md: Path = OUT_MD) -> dict:
    ma = mh.load_ma_molecules()
    mix = mh.load_ma_mixtures(ma)
    t = mix.table
    a, b = t.a_idx.to_numpy(), t.b_idx.to_numpy()
    obs = t.pleasantness_mix.to_numpy()

    tr = mh.load_keller_training(ma.cid, ma.stripped)  # the registered 376
    registered = pd.read_csv(mh.RESULTS_DIR / "mixture_hedonics_predictions.csv").set_index("mixture_id")
    reg_pred = registered.loc[t.mixture_id, "pred_ensemble"].to_numpy()

    # a. fitted baselines on the same training molecules
    base = fit_simple_baselines(simple_descriptors(keller_training_mols(tr)), tr.y)
    ma_x = simple_descriptors(ma.mols)
    mix_scores = {"ensemble": reg_pred}
    for name, predict in base.items():
        comp = predict(ma_x)
        mix_scores[name] = mh.mixture_rule(comp[a], comp[b])

    # b. near-duplicate subsets
    ma_feats = mh.featurize(ma.mols)
    sims = mh.tanimoto_matrix(ma_feats.ecfp, tr.features.ecfp).max(axis=1)
    subsets = {"all": np.ones(len(t), bool)}
    for c in NEAR_DUPLICATE_CUTOFFS:
        subsets[f"tanimoto_lt_{c}"] = (sims[a] < c) & (sims[b] < c)

    # c. prospective model (Ma and pool removed)
    pool = pr.load_pool()
    tr2 = mh.load_keller_training(set(ma.cid.tolist()) | set(pool.table.cid.tolist()), set(ma.stripped) | set(pool.table.stripped))
    ens2 = mh._ensemble_from_features(tr2.features, tr2.y, tr2.cids, rf_n_jobs=1)
    comp2 = ens2.predict(ma.mols).ensemble
    mix_scores["prospective_model"] = mh.mixture_rule(comp2[a], comp2[b])

    specs = [mh.StatSpec("ensemble", "spearman_mix", "ensemble"),
             mh.StatSpec("ridge", "spearman_mix", "ridge"), mh.StatSpec("rf8", "spearman_mix", "rf8"),
             mh.StatSpec("prospective_model", "spearman_mix", "prospective_model")]
    specs += [mh.StatSpec(f"ensemble|{k}", "spearman_mix", "ensemble", k) for k in subsets if k != "all"]
    data = mh.BootData(len(ma.cid), a, b, obs, obs > np.median(obs), mix_scores, subsets,
                       mh.component_alone_means(ma, mix), {})
    full, ones = np.ones(len(ma.cid), bool), np.ones(len(ma.cid), int)
    point = {s.name: mh.eval_stat(data, s, full, ones) for s in specs}
    boot = mh.cluster_bootstrap(data, specs, n_boot, mh.BOOT_SEED, n_jobs)

    def diff(name: str) -> dict:
        lo, hi, _ = mh.percentile_ci(boot["ensemble"] - boot[name])
        return {"estimate": point["ensemble"] - point[name], "ci95": [lo, hi]}

    result = {
        "label": LABEL,
        "n_mixtures": len(t), "training_registered": int(len(tr.y)), "training_prospective_model": int(len(tr2.y)),
        "bootstrap": {"n": n_boot, "seed": mh.BOOT_SEED, "unit": "molecule (mixture kept iff both components drawn)"},
        "a_baselines": {
            "descriptors": list(DESCRIPTORS),
            "ensemble": _ci(boot, "ensemble", point["ensemble"]),
            "ridge": {**_ci(boot, "ridge", point["ridge"]), "ensemble_minus_ridge": diff("ridge")},
            "rf8": {**_ci(boot, "rf8", point["rf8"]), "ensemble_minus_rf8": diff("rf8")},
        },
        "b_near_duplicate": {"max_tanimoto_to_training_median": float(np.median(sims)),
                             **{k: {"n_mixtures": int(m.sum()),
                                    **(_ci(boot, f"ensemble|{k}", point[f"ensemble|{k}"]))}
                                for k, m in subsets.items() if k != "all"}},
        "c_prospective_model_on_ma": {**_ci(boot, "prospective_model", point["prospective_model"]),
                                      "spearman_with_registered_model_mixture_predictions": mh.spearman(mix_scores["prospective_model"], reg_pred)},
    }
    if simulate:
        result["d_p1_cluster_inference"] = p1_simulation(pd.read_csv(FROZEN_CSV))
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    out_md.write_text(render_md(result))
    return result


def render_md(r: dict) -> str:
    f = lambda d: f"{d['estimate']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"  # noqa: E731
    a, b, c = r["a_baselines"], r["b_near_duplicate"], r["c_prospective_model_on_ma"]
    L = [f"# Robustness checks ({LABEL})", "",
         "Produced by `robustness.py` after adversarial review; none of this is registered. Spearman on the 198 Ma mixtures,",
         f"95% percentile CI from a molecule-cluster bootstrap ({r['bootstrap']['n']:,} resamples, registered seed).", "",
         "## a. Fitted simple baselines (same 376 Keller training molecules, equal-weight mixture rule)",
         "Features: " + ", ".join(a["descriptors"]) + ".", "",
         "| model | mixture rho [CI] | ensemble minus model [CI] |", "|---|---|---|",
         f"| registered ensemble | {f(a['ensemble'])} | |",
         f"| ridge (alpha 1, standardised) | {f(a['ridge'])} | {f(a['ridge']['ensemble_minus_ridge'])} |",
         f"| random forest (500 trees) | {f(a['rf8'])} | {f(a['rf8']['ensemble_minus_rf8'])} |", "",
         "## b. Near-duplicate sensitivity (ensemble H2 on mixtures with both components below the Tanimoto cutoff)",
         "| max ECFP4 Tanimoto to training | n mixtures | rho [CI] |", "|---|---|---|"]
    L += [f"| < {k.split('_')[-1]} | {v['n_mixtures']} | {f(v)} |" for k, v in b.items() if k.startswith("tanimoto")]
    L += ["", "## c. Prospective model (Ma and pool molecules removed, %d training molecules)" % r["training_prospective_model"],
          f"On the Ma mixtures: rho {f(c)}; Spearman with the registered model's mixture predictions "
          f"{c['spearman_with_registered_model_mixture_predictions']:.3f}.", ""]
    if "d_p1_cluster_inference" in r:
        d = r["d_p1_cluster_inference"]
        L += ["## d. P1 with molecule-clustered inference (simulation on the frozen 40 pairs, %d molecules)" % d["n_molecules"],
              "Rule: permutation p < 0.05 AND one-sided 95% cluster-bootstrap lower bound > 0. "
              f"{d['n_sims']} simulations, {d['n_boot_per_sim']} resamples each.", "",
              "| true rho (30-rater panel) | permutation alone | rule (power; type-I at 0) |", "|---|---|---|"]
        L += [f"| {k} | {v['p_permutation_alone']:.3f} | {v['p_permutation_and_cluster_lb']:.3f} |" for k, v in d["scenarios"].items()]
        L += ["", "Caveat: Gaussian copula, molecule-level errors independent across molecules; real errors will be messier."]
    return "\n".join(L) + "\n"


def main() -> None:
    print(render_md(run()))


if __name__ == "__main__":
    main()
