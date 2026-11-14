"""POST-REGISTRATION EXPLORATORY analyses on top of the registered mixture-hedonics results.

None of this was in PREREGISTRATION.md. Nothing here alters results/mixture_hedonics*.{json,csv};
those are read-only inputs. Every output is labelled EXPLORATORY.

  1. noise ceiling for H1/H2 from Ma 2021 per-rater data (split-half + Spearman-Brown)
  2. split-conformal abstention with molecule-disjoint calibration (+ Mondrian class-conditional)
  3. odorousness gate (Mayhew 2022 boundary; vapor pressure is an approximation, see below)
  4. power analysis for a prospective Ma-style binary-mixture panel

Run (repo root):  OMP_NUM_THREADS=1 PYTHONPATH=methods .venv/bin/python -m chocolate_anchor.exploratory
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from scipy.stats import binom, rankdata

RDLogger.DisableLog("rdApp.*")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
RESULTS_DIR = _HERE / "results"
LONG_CSV = _ROOT / "data" / "hedonics" / "mixture_pleasantness_ma2021_long.csv"
PRED_CSV = RESULTS_DIR / "mixture_hedonics_predictions.csv"
REGISTERED_JSON = RESULTS_DIR / "mixture_hedonics.json"
OUT_JSON = RESULTS_DIR / "exploratory.json"
OUT_MD = _HERE / "EXPLORATORY.md"

LABEL = "EXPLORATORY (post-registration; not part of PREREGISTRATION.md)"
N_SPLITS = 1000
SPLIT_SEED = 20261114
N_CONFORMAL_REPS = 200
CONFORMAL_SEED = 20261115
CONFORMAL_ALPHAS = (0.2, 0.1)
POWER_SEED = 20261116
N_POWER_SIMS = 2000
POWER_N_MIX = (20, 30, 40, 60)
POWER_N_RATERS = (20, 30, 40)
POWER_ALPHA = 0.05
N_NULL_DRAWS = 100_000
CONSERVATIVE_RHO = 0.3
N_JOBS = -1


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx @ ry) / math.sqrt((rx @ rx) * (ry @ ry)))


def spearman_brown(r_half: float) -> float:
    """Reliability of the full-length measure from the correlation of two halves (factor 2)."""
    return 2.0 * r_half / (1.0 + r_half)


# ======================================================================================
# 1. Noise ceiling
# ======================================================================================
class RaterData:
    """Per-rating rows: rater index, target (mixture or molecule) index, value."""

    def __init__(self, rater: np.ndarray, target: np.ndarray, value: np.ndarray, n_raters: int, n_targets: int):
        self.rater, self.target, self.value = rater, target, value
        self.n_raters, self.n_targets = n_raters, n_targets


def load_rater_data() -> tuple[RaterData, RaterData, list[str]]:
    """Mixture-level and component-alone-level rating tables from Ma's long per-rater file.

    Mixture rows: one per (rater, rep, mixture) pleasantness_mix. Component rows: each long row
    carries the rater's alone rating of both components, so each row contributes two component
    ratings (the registered H1 target, the mean over all such rows per molecule, is the same quantity).
    """
    d = pd.read_csv(LONG_CSV)
    rater_ids = np.sort(d.rater.unique())
    r_idx = pd.Series(np.arange(len(rater_ids)), index=rater_ids)[d.rater].to_numpy()
    mix_ids = list(dict.fromkeys(d.mixture_id))
    m_idx = pd.Series(np.arange(len(mix_ids)), index=mix_ids)[d.mixture_id].to_numpy()
    mixtures = RaterData(r_idx, m_idx, d.pleasantness_mix.to_numpy(float), len(rater_ids), len(mix_ids))

    cids = d.cids.str.split(";", expand=True).astype(int)
    mol_ids = np.sort(pd.unique(cids.to_numpy().ravel()))
    pos = pd.Series(np.arange(len(mol_ids)), index=mol_ids)
    comp = RaterData(
        np.concatenate([r_idx, r_idx]),
        np.concatenate([pos[cids[0]].to_numpy(), pos[cids[1]].to_numpy()]),
        np.concatenate([d.pleas_A_alone.to_numpy(float), d.pleas_B_alone.to_numpy(float)]),
        len(rater_ids), len(mol_ids),
    )
    return mixtures, comp, mix_ids


def _half_means(data: RaterData, w_rater_half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean rating per target over the raters selected (weighted) by w_rater_half; NaN if none."""
    w = w_rater_half[data.rater]
    num = np.bincount(data.target, weights=w * data.value, minlength=data.n_targets)
    den = np.bincount(data.target, weights=w, minlength=data.n_targets)
    out = np.full(data.n_targets, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out, den


def split_half_reliability(data: RaterData, rng: np.random.Generator, rater_weight: np.ndarray | None = None) -> float:
    """One random split of the raters into halves -> Spearman between half means -> Spearman-Brown."""
    perm = rng.permutation(data.n_raters)
    wa = np.zeros(data.n_raters)
    wb = np.zeros(data.n_raters)
    wa[perm[: data.n_raters // 2]] = 1.0
    wb[perm[data.n_raters // 2:]] = 1.0
    if rater_weight is not None:
        wa, wb = wa * rater_weight, wb * rater_weight
    ma, _ = _half_means(data, wa)
    mb, _ = _half_means(data, wb)
    ok = ~np.isnan(ma) & ~np.isnan(mb)
    return spearman_brown(spearman(ma[ok], mb[ok]))


def noise_ceiling_one(data: RaterData, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rel = np.array([split_half_reliability(data, rng) for _ in range(N_SPLITS)])
    ceil = np.sqrt(np.clip(rel, 0, 1))
    return {
        "reliability_full_panel_sb": {"mean": float(rel.mean()), "split_ci95": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))]},
        "ceiling_rho_sqrt_reliability": {"mean": float(ceil.mean()), "split_ci95": [float(np.percentile(ceil, 2.5)), float(np.percentile(ceil, 97.5))]},
        "n_splits": N_SPLITS,
        "ci_note": "CI = 2.5/97.5 percentiles over random splits only: it reflects split randomness, not sampling uncertainty of the 60-rater panel (a rater bootstrap was tried and rejected: resampled panels hold ~63% unique raters, which biases reliability downward)",
    }


def noise_ceiling(registered: dict) -> dict:
    mixtures, comp, _ = load_rater_data()
    per_mix = pd.read_csv(LONG_CSV).groupby("mixture_id").rater.nunique()
    out: dict = {
        "label": LABEL,
        "method": "random halves of the 60 raters (each mixture/molecule is rated by the raters of each half that rated it); half-panel means; "
                  "Spearman across targets; Spearman-Brown to the full panel gives reliability R; the ceiling for the Spearman of a noise-free "
                  "predictor with the observed panel mean is sqrt(R).",
        "raters": mixtures.n_raters, "raters_per_mixture_min_max": [int(per_mix.min()), int(per_mix.max())],
    }
    mix = noise_ceiling_one(mixtures, SPLIT_SEED)
    cmp_ = noise_ceiling_one(comp, SPLIT_SEED + 10_000)
    h2, oracle, h1 = registered["H2"]["estimate"], registered["A3_oracle_measured_components"]["mixture_spearman"]["estimate"], registered["H1"]["estimate"]
    c_m, c_c = mix["ceiling_rho_sqrt_reliability"], cmp_["ceiling_rho_sqrt_reliability"]

    def frac(x: float, c: dict) -> dict:
        return {"rho": x, "fraction_of_ceiling": x / c["mean"],
                "fraction_range_over_ceiling_split_ci": [x / c["split_ci95"][1], x / c["split_ci95"][0]]}

    mix["H2"], mix["oracle_A3"] = frac(h2, c_m), frac(oracle, c_m)
    cmp_["H1"] = frac(h1, c_c)
    out["mixtures"], out["components"] = mix, cmp_
    out["note"] = ("Fractions use the registered point estimates; their own sampling CIs are not propagated. The oracle uses the same raters' "
                   "alone ratings as the target, so its noise is correlated with the target and it can exceed the independent-noise ceiling.")
    return out


# ======================================================================================
# 2. Conformal abstention
# ======================================================================================
def ecdf_rank(reference: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mid-rank empirical CDF of x against `reference`, in [0, 1]."""
    ref = np.sort(reference)
    lo, hi = np.searchsorted(ref, x, side="left"), np.searchsorted(ref, x, side="right")
    return (lo + hi) / (2.0 * len(ref))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal q-hat: the ceil((n+1)(1-alpha))/n empirical quantile of the calibration scores.
    If that level exceeds 1 the set is unbounded (inf)."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))  # the k-th smallest score is the ceil((n+1)(1-alpha))/n empirical quantile
    if k > n:
        return math.inf
    return float(np.sort(scores)[k - 1])


def conformal_rep(cal_obs, cal_pred, test_obs, test_pred, test_confident, alpha: float, liked_value: float) -> dict:
    """One calibration/test pair. Common rank scale: observed and predicted values are mid-rank-ECDF
    transformed against the CALIBRATION set's observed / predicted values respectively; the liked threshold
    on that scale is the ECDF of the registered median among the calibration observations."""
    t = float(ecdf_rank(cal_obs, np.array([liked_value]))[0])
    cal_o, cal_p = ecdf_rank(cal_obs, cal_obs), ecdf_rank(cal_pred, cal_pred)
    te_o, te_p = ecdf_rank(cal_obs, test_obs), ecdf_rank(cal_pred, test_pred)
    truth_liked = test_obs > liked_value
    scores = np.abs(cal_o - cal_p)
    q = conformal_quantile(scores, alpha)
    lo, hi = te_p - q, te_p + q
    covered = np.abs(te_o - te_p) <= q
    width = np.minimum(hi, 1.0) - np.maximum(lo, 0.0)

    def decide(liked: np.ndarray, disliked: np.ndarray) -> dict:
        decided = liked ^ disliked
        correct = (liked & truth_liked) | (disliked & ~truth_liked)
        return {"n_decided": int(decided.sum()), "n_correct": int((decided & correct).sum()), "n": len(truth_liked)}

    out = {"marginal": {"coverage": float(covered.mean()), "mean_width": float(width.mean()),
                        **decide(lo > t, hi <= t)}}
    # Mondrian (class-conditional): q per observed class; label k is in the set if k's interval reaches k's region.
    cal_liked = cal_obs > liked_value
    q_l = conformal_quantile(scores[cal_liked], alpha)
    q_d = conformal_quantile(scores[~cal_liked], alpha)
    in_l, in_d = te_p + q_l > t, te_p - q_d <= t
    err = np.abs(te_o - te_p)
    cov_l = float((err[truth_liked] <= q_l).mean()) if truth_liked.any() else float("nan")
    cov_d = float((err[~truth_liked] <= q_d).mean()) if (~truth_liked).any() else float("nan")
    out["mondrian"] = {"coverage_liked": cov_l, "coverage_disliked": cov_d, "n_empty_set": int((~in_l & ~in_d).sum()),
                       **decide(in_l & ~in_d, in_d & ~in_l)}
    # H3 disagreement rule on the same test mixtures: decide only when the four model families agree
    # (registered 'confident' flag); direction = predicted rank vs the same threshold.
    out["h3_rule"] = decide((te_p > t) & test_confident, (te_p <= t) & test_confident)
    out["no_abstention"] = decide(te_p > t, te_p <= t)
    return out


def conformal_analysis() -> dict:
    pred = pd.read_csv(PRED_CSV)
    cids = pred.cids.str.split(";", expand=True).astype(int).to_numpy()
    mols = np.unique(cids)
    obs, ens, conf = pred.observed.to_numpy(float), pred.pred_ensemble.to_numpy(float), pred.confident.to_numpy(bool)
    liked_value = float(np.median(obs))
    rng = np.random.default_rng(CONFORMAL_SEED)
    reps = []
    for _ in range(N_CONFORMAL_REPS):
        half1 = set(rng.permutation(mols)[: len(mols) // 2].tolist())
        in1 = np.isin(cids, list(half1))
        cal, test = in1.all(axis=1), (~in1).all(axis=1)
        assert not (cal & test).any()
        reps.append((cal, test))
    n_cal = np.array([c.sum() for c, _ in reps]); n_test = np.array([t.sum() for _, t in reps])
    result: dict = {"label": LABEL, "n_molecules": len(mols), "n_mixtures": len(pred), "reps": N_CONFORMAL_REPS,
                    "n_cal_mean_min_max": [float(n_cal.mean()), int(n_cal.min()), int(n_cal.max())],
                    "n_test_mean_min_max": [float(n_test.mean()), int(n_test.min()), int(n_test.max())],
                    "n_dropped_straddling_mean": float(len(pred) - (n_cal + n_test).mean()),
                    "liked_threshold_value": liked_value,
                    "rank_scale": "mid-rank ECDF of observed (resp. predicted) values against the calibration set's observed (resp. predicted) values; "
                                  "score = |obs rank - pred rank|; interval = pred rank +/- q-hat, clipped to [0,1] for width",
                    "predictions": "registered ensemble predictions (models trained on Keller with every Ma molecule removed), so the split concerns only calibration/test"}
    for alpha in CONFORMAL_ALPHAS:
        reps_out = [conformal_rep(obs[c], ens[c], obs[t], ens[t], conf[t], alpha, liked_value) for c, t in reps]
        block: dict = {"target_coverage": 1 - alpha}
        for name in ("marginal", "mondrian", "h3_rule", "no_abstention"):
            rows = [r[name] for r in reps_out]
            dec, cor, n = (np.array([r[k] for r in rows]) for k in ("n_decided", "n_correct", "n"))
            entry = {"decision_rate_pooled": float(dec.sum() / n.sum()),
                     "accuracy_on_decided_pooled": float(cor.sum() / dec.sum()) if dec.sum() else None,
                     "accuracy_on_decided_rep_mean_sd": ([float(np.mean(cor[dec > 0] / dec[dec > 0])), float(np.std(cor[dec > 0] / dec[dec > 0]))] if (dec > 0).any() else None),
                     "reps_with_any_decision": int((dec > 0).sum()), "n_decided_total": int(dec.sum())}
            if name == "marginal":
                cov = np.array([r["coverage"] for r in rows]); w = np.array([r["mean_width"] for r in rows])
                entry |= {"coverage_mean": float(cov.mean()), "coverage_sd": float(cov.std()),
                          "coverage_5_95pct": [float(np.percentile(cov, 5)), float(np.percentile(cov, 95))],
                          "mean_interval_width": float(w.mean())}
            if name == "mondrian":
                entry |= {"coverage_liked_mean": float(np.nanmean([r["coverage_liked"] for r in rows])),
                          "coverage_disliked_mean": float(np.nanmean([r["coverage_disliked"] for r in rows])),
                          "empty_set_rate": float(sum(r["n_empty_set"] for r in rows) / n.sum())}
            block[name] = entry
        result[f"alpha_{alpha}"] = block
    return result


# ======================================================================================
# 3. Odorousness gate (Mayhew et al. 2022)
# ======================================================================================
# Verified (METHODS_TO_INTEGRATE.md section 2, mayhew2022 OPENED-FULL): odorous side of the 50% boundary,
# VP in mmHg, logs assumed base 10:
#   room-temperature liquids/solids: logP >  -1.72 log VP - 9.10
#   gases:                           logP <  -1.61 log VP + 8.17
# Mayhew's logP and VP come from EPI Suite. Here logP = RDKit Crippen (an approximation) and the
# vapor pressure MUST be supplied from a measured or EPI Suite value: no VP is estimated, because an
# estimator would need values we have not sourced. Gas = VP >= 760 mmHg (stated assumption).
MAYHEW_SOLID_LIQUID = (-1.72, -9.10)
MAYHEW_GAS = (-1.61, 8.17)
GAS_LOG_VP = math.log10(760.0)


def odorous_side(logp: float, log_vp: float) -> tuple[bool, str]:
    """Exact Mayhew 2022 boundary; gas branch when log_vp >= log10(760)."""
    if log_vp >= GAS_LOG_VP:
        bound = MAYHEW_GAS[0] * log_vp + MAYHEW_GAS[1]
        return logp < bound, f"gas boundary: logP {logp:.2f} {'<' if logp < bound else '>='} {bound:.2f}"
    bound = MAYHEW_SOLID_LIQUID[0] * log_vp + MAYHEW_SOLID_LIQUID[1]
    return logp > bound, f"liquid/solid boundary: logP {logp:.2f} {'>' if logp > bound else '<='} {bound:.2f}"


def is_likely_odorous(smiles: str, vapor_pressure_mmhg: float) -> tuple[bool, str]:
    """Mayhew 2022 transport-feature gate with a supplied (measured or EPI Suite) vapor pressure.
    logP is RDKit Crippen, an approximation of the paper's EPI Suite KOWWIN value."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    if vapor_pressure_mmhg <= 0:
        raise ValueError("vapor pressure must be positive")
    log_vp = math.log10(vapor_pressure_mmhg)
    ok, why = odorous_side(float(Crippen.MolLogP(mol)), log_vp)
    return ok, f"{why} (logVP {log_vp:.2f}, supplied VP)"


GATE_NOTE = ("Gate implemented (mayhew2022 boundary equations, verified) but not validated or applied: it needs a "
             "sourced vapor pressure per molecule, which this repo does not have. Not needed for the prospective "
             "study: every candidate is a listed flavoring, i.e. odorous in use.")


# ======================================================================================
# 4. Power analysis for a prospective Ma-style panel
# ======================================================================================
def fit_variance_components(mixtures: RaterData) -> dict:
    """y = mu_i + b_r + e on all rows (alternating least squares). Returns sigma_mu, sigma_b, sigma_e and the
    empirical residuals, mean-centred, used for resampling."""
    mu = np.zeros(mixtures.n_targets); b = np.zeros(mixtures.n_raters)
    t, r, y = mixtures.target, mixtures.rater, mixtures.value
    n_t = np.bincount(t, minlength=mixtures.n_targets); n_r = np.bincount(r, minlength=mixtures.n_raters)
    for _ in range(200):
        mu = np.bincount(t, weights=y - b[r], minlength=mixtures.n_targets) / n_t
        b = np.bincount(r, weights=y - mu[t], minlength=mixtures.n_raters) / n_r
        b -= b.mean()
    resid = y - mu[t] - b[r]
    dof = len(y) - mixtures.n_targets - mixtures.n_raters + 1
    sigma_e2 = float((resid ** 2).sum() / dof)
    sigma_mu2 = float(mu.var(ddof=1) - np.mean(sigma_e2 / n_t))
    return {"mu_hat": mu, "resid": resid * math.sqrt(len(y) / dof), "sigma_e": math.sqrt(sigma_e2), "sigma_b": float(b.std(ddof=1)),
            "sigma_mu": math.sqrt(max(sigma_mu2, 0.0)), "shrink": math.sqrt(max(sigma_mu2, 0.0) / mu.std(ddof=1))}


def spearman_critical(n: int, alpha: float, seed: int) -> float:
    """One-sided Monte-Carlo critical value of Spearman rho under independence at sample size n."""
    rng = np.random.default_rng(seed)
    x = np.arange(1, n + 1, dtype=float); x -= x.mean()
    perms = np.argsort(rng.random((N_NULL_DRAWS, n)), axis=1).astype(float); perms -= perms.mean(axis=1, keepdims=True)
    rho = (perms * x).sum(axis=1) / (np.linalg.norm(perms, axis=1) * np.linalg.norm(x))
    return float(np.quantile(rho, 1 - alpha))


def binom_critical(n_decided: int, alpha: float) -> int:
    """Smallest k with P(Binomial(n, 0.5) >= k) <= alpha (accuracy > 0.5 test); n+1 if none."""
    for k in range(n_decided + 1):
        if binom.sf(k - 1, n_decided, 0.5) <= alpha:
            return k
    return n_decided + 1


def _rowwise_rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a, axis=1), axis=1).astype(float)


def simulate_cell(vc: dict, rho_latent: float, n_mix: int, n_raters: int, decided_fracs: Sequence[float], crit: float, seed: int) -> dict:
    """Power for one (n_mix, n_raters) cell.
    True mixture means mu_i: Ma's shrunken estimated means resampled with replacement.
    Predictions: Gaussian copula with Spearman rho_latent against mu (Pearson 2 sin(pi rho/6) on normal scores).
    Panel mean: mu + mean of n_raters residuals resampled from Ma's empirical rater x mixture residuals
    (every simulated rater rates every mixture, so rater offsets add a constant and do not affect ranks)."""
    from scipy.special import ndtri

    rng = np.random.default_rng(seed)
    S = N_POWER_SIMS
    mu = rng.choice(vc["mu_hat"] - vc["mu_hat"].mean(), size=(S, n_mix)) * vc["shrink"]
    mu = mu + rng.normal(0, 1e-9, mu.shape)  # break resampling ties
    z = ndtri((_rowwise_rank(mu) + 0.5) / n_mix)
    r = 2 * math.sin(math.pi * rho_latent / 6)
    pred = r * z + math.sqrt(1 - r * r) * rng.normal(size=z.shape)
    noise = rng.choice(vc["resid"], size=(S, n_mix, n_raters)).mean(axis=2)
    panel = mu + noise
    rp, ro = _rowwise_rank(pred), _rowwise_rank(panel)
    rp -= rp.mean(1, keepdims=True); ro -= ro.mean(1, keepdims=True)
    rho = (rp * ro).sum(1) / np.sqrt((rp ** 2).sum(1) * (ro ** 2).sum(1))
    out = {"power_spearman": float((rho > crit).mean()), "mean_observed_rho": float(rho.mean())}
    liked = ro > np.median(ro, axis=1, keepdims=True)  # above the panel median, as in H3
    ext = np.abs(rp / n_mix)  # distance of predicted rank from the median: decided = the most extreme share
    for d in decided_fracs:
        k = max(int(round(d * n_mix)), 1)
        top = np.argsort(-ext, axis=1)[:, :k]
        correct = np.take_along_axis((rp > 0) == liked, top, axis=1).sum(1)
        out[f"power_accuracy_gt_half@decided={d:.2f}"] = float((correct >= binom_critical(k, POWER_ALPHA)).mean())
        out[f"n_decided@{d:.2f}"] = k
    return out


def power_analysis(registered: dict, conformal: dict, noise: dict) -> dict:
    mixtures, _, _ = load_rater_data()
    vc = fit_variance_components(mixtures)
    r_split = noise["mixtures"]["reliability_full_panel_sb"]["mean"]
    r_design = vc["sigma_mu"] ** 2 / (vc["sigma_mu"] ** 2 + vc["sigma_e"] ** 2 / 30)
    h2 = registered["H2"]["estimate"]
    decided = sorted({round(conformal["alpha_0.2"]["marginal"]["decision_rate_pooled"], 3),
                      round(registered_confident_rate(), 3)})
    out: dict = {"label": LABEL, "n_sims_per_cell": N_POWER_SIMS, "alpha_one_sided": POWER_ALPHA,
                 "variance_components_from_Ma": {k: vc[k] for k in ("sigma_mu", "sigma_b", "sigma_e")},
                 "reliability_ma_split_half": r_split, "reliability_design_matched_30_raters": r_design,
                 "decided_fractions_used": decided, "scenarios": {}}
    crit = {n: spearman_critical(n, POWER_ALPHA, POWER_SEED + n) for n in POWER_N_MIX}
    out["spearman_critical_values"] = crit
    for disatt_name, rel in (("ma_split_half", r_split), ("design_matched", r_design)):
        for target in (h2, CONSERVATIVE_RHO):
            rho_lat = min(target / math.sqrt(rel), 0.99)
            cells = Parallel(n_jobs=N_JOBS)(
                delayed(simulate_cell)(vc, rho_lat, n, m, decided, crit[n], POWER_SEED + 100 * n + m)
                for n in POWER_N_MIX for m in POWER_N_RATERS)
            keys = [(n, m) for n in POWER_N_MIX for m in POWER_N_RATERS]
            out["scenarios"][f"{disatt_name}|target_rho={target:.3f}"] = {
                "latent_rho": rho_lat, "cells": {f"n_mix={n},n_raters={m}": c for (n, m), c in zip(keys, cells)}}
    out["assumptions"] = [
        "Target rho is the expected Spearman with a Ma-sized (30-rater) panel mean; disattenuated to a latent rho by sqrt(reliability), "
        "then re-attenuated for each simulated panel size. 'ma_split_half' uses Ma's split-half reliability (rater offsets inflate Ma's noise); "
        "'design_matched' uses the variance-component reliability with offsets cancelled (every rater rates every mixture).",
        "Rater x mixture noise is resampled from Ma's empirical residuals (independent across raters), true means from Ma's shrunken estimated means.",
        "Predictions are a Gaussian copula with the stated latent Spearman; real prediction errors are not Gaussian and correlate across mixtures sharing components.",
        "Mixtures are treated as independent (Ma's share components, so real power is lower). Spearman p from a Monte-Carlo null; classifier test is exact binomial.",
        "Classifier: 'decided' = the most extreme-predicted share of mixtures (conformal decision rate and H3 confident rate), liked = above the panel median.",
    ]
    return out


def registered_confident_rate() -> float:
    return float(pd.read_csv(PRED_CSV).confident.mean())


# ======================================================================================
# report
# ======================================================================================
def write_md(res: dict) -> str:
    dec = res["power"]["decided_fractions_used"]
    nc, cf, gv, pw = res["noise_ceiling"], res["conformal"], res["odorousness_gate"], res["power"]
    m, c = nc["mixtures"], nc["components"]
    f = lambda x, d=2: f"{x:.{d}f}"  # noqa: E731
    L = ["# EXPLORATORY analyses (post-registration)", "",
         "**Not pre-registered.** Run after the registered results (`RESULTS_MIXTURE_HEDONICS.md`), which are untouched and read-only here. "
         "Code `exploratory.py`, numbers `results/exploratory.json`. Citation keys are in `docs/literature/REFERENCES.bib`.", "",
         "## 1. Noise ceiling (Keller2017dream-style test-retest logic; Spearman-Brown)",
         f"Method: 1000 random halves of Ma's 60 raters, half-panel means, Spearman over targets, Spearman-Brown to the full panel (R); ceiling = sqrt(R) "
         f"(max rho of a noise-free predictor). Cites `keller2017dream` (noise ceiling as upper limit), `ma2021data` (per-rater data).",
         f"- Mixtures: R = {f(m['reliability_full_panel_sb']['mean'])} (split CI {f(m['reliability_full_panel_sb']['split_ci95'][0])}-{f(m['reliability_full_panel_sb']['split_ci95'][1])}); "
         f"ceiling rho = {f(m['ceiling_rho_sqrt_reliability']['mean'])} (split CI {f(m['ceiling_rho_sqrt_reliability']['split_ci95'][0])}-{f(m['ceiling_rho_sqrt_reliability']['split_ci95'][1])}).",
         f"  H2 rho 0.490 = {f(100 * m['H2']['fraction_of_ceiling'], 0)}% of ceiling; oracle A3 0.897 = {f(100 * m['oracle_A3']['fraction_of_ceiling'], 0)}%.",
         f"- Components: R = {f(c['reliability_full_panel_sb']['mean'])} (split CI {f(c['reliability_full_panel_sb']['split_ci95'][0])}-{f(c['reliability_full_panel_sb']['split_ci95'][1])}); "
         f"ceiling rho = {f(c['ceiling_rho_sqrt_reliability']['mean'])} (split CI {f(c['ceiling_rho_sqrt_reliability']['split_ci95'][0])}-{f(c['ceiling_rho_sqrt_reliability']['split_ci95'][1])}); H1 rho 0.640 = {f(100 * c['H1']['fraction_of_ceiling'], 0)}% of ceiling.",
         "- CIs are split-to-split spread only (excludes panel sampling uncertainty). Raters rate different subsets (~30 of 60 per mixture): halves are formed over raters globally; each target uses whichever raters of the half rated it.", "",
         "## 2. Split-conformal abstention (Angelopoulos & Bates, `angelopoulos2023`)",
         f"Quantile q-hat = ceil((n+1)(1-a))/n quantile (verified, opened). Score = |obs rank - pred rank| on the calibration-ECDF rank scale; the absolute-residual "
         f"regression recipe itself was NOT located in the opened text (VERIFICATION.md), so it is the standard choice, not a verified citation. "
         f"{cf['reps']} molecule-disjoint splits (72 molecules halved; calibration mixtures have both components in half 1, test both in half 2; "
         f"n_cal ~{f(cf['n_cal_mean_min_max'][0], 0)}, n_test ~{f(cf['n_test_mean_min_max'][0], 0)}, ~{f(cf['n_dropped_straddling_mean'], 0)} straddling dropped). "
         "Decision: liked = interval wholly above the median threshold, disliked = wholly below, else abstain. Mondrian = class-conditional q-hat per observed class "
         "(`angelopoulos2023` class-conditional recipe, opened; `vovk2013conditional` not opened), label kept if its interval reaches its region.", "",
         "| a | cover (mean, 5-95%) | width | 3-way decided / acc | Mondrian decided / acc (cover liked/disl.) | H3 rule decided / acc | no abstention acc |", "|---|---|---|---|---|---|---|"]
    for a in CONFORMAL_ALPHAS:
        b = cf[f"alpha_{a}"]
        g = lambda k: (f"{f(100 * b[k]['decision_rate_pooled'], 0)}% / " + (f(b[k]['accuracy_on_decided_pooled']) if b[k]['accuracy_on_decided_pooled'] is not None else "n/a"))  # noqa: E731
        L.append(f"| {a} (target {f(1 - a)}) | {f(b['marginal']['coverage_mean'])} ({f(b['marginal']['coverage_5_95pct'][0])}-{f(b['marginal']['coverage_5_95pct'][1])}) | "
                 f"{f(b['marginal']['mean_interval_width'])} | {g('marginal')} | {g('mondrian')} ({f(b['mondrian']['coverage_liked_mean'])}/{f(b['mondrian']['coverage_disliked_mean'])}) | "
                 f"{g('h3_rule')} | {f(b['no_abstention']['accuracy_on_decided_pooled'])} |")
    L += ["Coverage is marginal over test mixtures that share components; exchangeability is only approximate.", "",
          "## 3. Odorousness gate (`mayhew2022`)",
          gv["status"]]
    L += ["", 
          "## 4. Power for a prospective Ma-style panel (simulation; assumptions in JSON)",
          f"Resamples Ma's data structure (sigma_mu {f(pw['variance_components_from_Ma']['sigma_mu'])}, sigma_e {f(pw['variance_components_from_Ma']['sigma_e'])}; "
          "Ma's empirical rater residuals), Gaussian-copula predictions, alpha 0.05 one-sided, 2000 sims/cell. Targets: registered H2 rho 0.49 and 0.30 "
          "as the expected Spearman at a 30-rater panel (disattenuated by Ma's split-half R). Cells: Spearman power by n_mixtures (rows) x n_raters 20/30/40; "
          f"classifier-accuracy power (>0.5 on decided) in brackets at decided share {dec[0]:.2f} (conformal 3-way) | {dec[-1]:.2f} (H3 confident rate)."]
    for sc, v in pw["scenarios"].items():
        if not sc.startswith("ma_split_half"):
            continue
        L.append(f"- {sc.split('|')[1]}, latent rho {f(v['latent_rho'])}:")
        for n in POWER_N_MIX:
            L.append("  n=%d: " % n + "; ".join(
                f"{m_}: {f(v['cells'][f'n_mix={n},n_raters={m_}']['power_spearman'], 2)} [{f(v['cells'][f'n_mix={n},n_raters={m_}'][f'power_accuracy_gt_half@decided={dec[0]:.2f}'], 2)}|{f(v['cells'][f'n_mix={n},n_raters={m_}'][f'power_accuracy_gt_half@decided={dec[-1]:.2f}'], 2)}]" for m_ in POWER_N_RATERS))
    L.append("The design_matched disattenuation variant is in the JSON.")
    return "\n".join(L) + "\n"


def main() -> None:
    registered = json.loads(REGISTERED_JSON.read_text())
    noise = noise_ceiling(registered)
    conformal = conformal_analysis()
    gate = {"label": LABEL, "citation": "mayhew2022", "status": GATE_NOTE}
    power = power_analysis(registered, conformal, noise)
    res = {"label": LABEL, "noise_ceiling": noise, "conformal": conformal, "odorousness_gate": gate, "power": power}
    OUT_JSON.write_text(json.dumps(res, indent=2))
    OUT_MD.write_text(write_md(res))
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
