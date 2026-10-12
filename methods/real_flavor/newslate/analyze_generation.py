"""Analysis + Figure 1 for run_generation.py output (see that file)."""
import json, os, sys, itertools
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.abspath(os.path.join(HERE, "..", "results", "newslate"))
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "..", "figures"))
TERMS = ["taste", "composition", "novelty", "mixture"]


def main():
    import jspec
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem import Descriptors
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold
    import recipe_classifier as rc
    raw = json.load(open(os.path.join(OUT, "generation_raw.json")))
    ctx = jspec.Context("oof")
    pool = ctx.pool
    # independent taste model (descriptor GBM) -> T_alt, OOF for known molecules
    D = np.vstack([np.clip(np.nan_to_num(np.array(list(Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(p["smiles"]))).values() if False else list(Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(p["smiles"])).values()), dtype=float), posinf=0, neginf=0), -1e6, 1e6) for p in pool])
    k = np.where(ctx.is_known)[0]; y = np.array([pool[i]["pleasantness"] for i in k])
    mk = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    talt = mk().fit(D[k], y).predict(D)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(k):
        talt[k[te]] = mk().fit(D[k[tr]], y[tr]).predict(D[k[te]])
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(p["smiles"])) for p in pool]
    clf = rc.RecipeClassifier().fit()
    rows = []
    for r in tqdm(raw, desc="metrics"):
        for rank, rec in enumerate(r["recipes"]):
            c = rec["combo"]; t = rec["terms_oof"]
            smis = [pool[i]["smiles"] for i in c]
            div = np.mean([1 - DataStructs.TanimotoSimilarity(fps[a], fps[b]) for a, b in itertools.combinations(c, 2)])
            rows.append({"cfg": r["cfg"], "seed": r["seed"], "rank": rank, "T": t["T"], "C": t["C"], "N": t["N"], "M": t["M"],
                         "T_alt": float(np.mean(talt[c]) / 100), "P_good": float(clf.score(smis)),
                         "frac_denovo": float(np.mean(~ctx.is_known[c])), "diversity": float(div), "combo": c})
    json.dump(rows, open(os.path.join(OUT, "generation_recipe_metrics.json"), "w"), default=float)
    METS = ["T", "C", "N", "M", "T_alt", "P_good", "frac_denovo", "diversity"]
    seedmean = {}
    for cfg in {r["cfg"] for r in rows}:
        for sd in {r["seed"] for r in rows if r["cfg"] == cfg}:
            sel = [r for r in rows if r["cfg"] == cfg and r["seed"] == sd]
            seedmean[(cfg, sd)] = {m: float(np.mean([r[m] for r in sel])) for m in METS} | {"n_recipes": len(sel),
                                   "union": sorted({i for r in sel for i in r["combo"]}),
                                   "top": sorted(sel[0]["combo"])}
    res = {}
    rng = np.random.default_rng(0)

    def boot_mean_diff(a, b, n=4000):
        a, b = np.array(a), np.array(b); v = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
        return float(a.mean() - b.mean()), [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    def paired(a_cfg, b_cfg, seeds):
        out = {}
        for m in METS:
            d = np.array([seedmean[(a_cfg, s)][m] - seedmean[(b_cfg, s)][m] for s in seeds])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
            out[m] = {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      "mean_A": float(np.mean([seedmean[(a_cfg, s)][m] for s in seeds])), "mean_B": float(np.mean([seedmean[(b_cfg, s)][m] for s in seeds]))}
        return out
    jac = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))
    # --- (b): additive (1110) vs full (1111), seeds 0-9
    S10 = list(range(10))
    res["b_full_minus_additive_paired_by_seed"] = paired("1111", "1110", S10)
    res["b_union_set_jaccard_additive_vs_full_same_seed"] = float(np.mean([jac(seedmean[("1110", s)]["union"], seedmean[("1111", s)]["union"]) for s in S10]))
    res["b_union_set_jaccard_noise_floor_full_vs_full_other_seed"] = float(np.mean([jac(seedmean[("1111", a)]["union"], seedmean[("1111", b)]["union"]) for a in S10 for b in S10 if a != b]))
    res["b_union_set_jaccard_noise_floor_additive_vs_additive_other_seed"] = float(np.mean([jac(seedmean[("1110", a)]["union"], seedmean[("1110", b)]["union"]) for a in S10 for b in S10 if a != b]))
    res["b_top_recipe_jaccard_additive_vs_full_same_seed"] = float(np.mean([jac(seedmean[("1110", s)]["top"], seedmean[("1111", s)]["top"]) for s in S10]))
    res["b_legacy_insample_vs_oof_additive"] = paired("legacy_insample", "1110", S10)
    res["b_all_pool_vs_known_only_full"] = paired("1111", "full_known_only", list(range(5)))
    res["b_all_pool_vs_known_only_additive"] = paired("1110", "additive_known_only", list(range(5)))
    # --- (c): 2^4 factorial main effects (seeds 0-4)
    cfgs = ["".join(map(str, b)) for b in itertools.product([0, 1], repeat=4)]
    Xd, Yd = [], {m: [] for m in METS}
    for cfg in cfgs:
        for s in range(5):
            Xd.append([1.0] + [float(ch) for ch in cfg])
            for m in METS: Yd[m].append(seedmean[(cfg, s)][m])
    Xd = np.array(Xd); XtXi = np.linalg.inv(Xd.T @ Xd); eff = {}
    for m in METS:
        yv = np.array(Yd[m]); beta = XtXi @ Xd.T @ yv; resid = yv - Xd @ beta
        s2 = resid @ resid / (len(yv) - 5); se = np.sqrt(np.diag(s2 * XtXi)); sd = yv.std()
        eff[m] = {t: {"effect": float(beta[i + 1]), "ci95": [float(beta[i + 1] - 1.99 * se[i + 1]), float(beta[i + 1] + 1.99 * se[i + 1])],
                      "standardized": float(beta[i + 1] / sd)} for i, t in enumerate(TERMS)} | {"sd_across_runs": float(sd)}
    res["c_factorial_main_effects_on_output_metrics"] = eff
    res["c_config_means"] = {cfg: {m: float(np.mean([seedmean[(cfg, s)][m] for s in range(5)])) for m in METS} for cfg in cfgs}
    # drop-one Jaccard vs full, vs seed noise floor
    nf = res["b_union_set_jaccard_noise_floor_full_vs_full_other_seed"]
    drop = {}
    for t, cfg in zip(TERMS, ["0111", "1011", "1101", "1110"]):
        j = [jac(seedmean[(cfg, s)]["union"], seedmean[("1111", s)]["union"]) for s in range(5)]
        bs = [np.mean(np.array(j)[rng.integers(0, 5, 5)]) for _ in range(4000)]
        drop[t] = {"jaccard_union_vs_full_same_seed": float(np.mean(j)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res["c_drop_one_jaccard_vs_full"] = drop; res["c_noise_floor_jaccard"] = nf
    nul = [jac(seedmean[("0000", s)]["union"], seedmean[("1111", s)]["union"]) for s in range(5)]
    res["c_random_feasible_baseline_0000_jaccard_vs_full"] = float(np.mean(nul))
    # --- term scales on random feasible blends (nominal influence = weight x sd)
    rr = np.random.default_rng(1); vals = {t: [] for t in "TCNM"}
    n_try = 0
    while len(vals["T"]) < 2000:
        cb = tuple(rr.choice(ctx.elig, 10, replace=False))
        if ctx.N(cb) < 0.5: continue
        tt = ctx.terms(cb)
        for t in "TCNM": vals[t].append(tt[t])
    w = jspec.SPEC["weights"]
    res["term_scales_random_feasible_blends"] = {t: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "weighted_sd": float(np.std(v) * w[n]),}
                                                 for (t, v), n in zip(vals.items(), TERMS)}
    # --- non-additivity of M on blends
    single = np.array([ctx.M((i,)) for i in tqdm(range(ctx.n), desc="M singletons")])
    def nonadd(combos):
        Mv = np.array([ctx.M(c) for c in combos]); Ma = np.array([single[list(c)].mean() for c in combos])
        from scipy.stats import spearmanr
        return {"n": len(combos), "mean_M": float(Mv.mean()), "mean_M_additive_aggregate": float(Ma.mean()),
                "mean_M_minus_additive": float((Mv - Ma).mean()), "mean_abs_gap": float(np.abs(Mv - Ma).mean()),
                "spearman_M_vs_additive": float(spearmanr(Mv, Ma).correlation), "sd_M": float(Mv.std())}
    rc_ = [tuple(rr.choice(ctx.elig, 10, replace=False)) for _ in range(300)]
    res["M_nonadditivity_random_blends"] = nonadd(rc_)
    res["M_nonadditivity_generated_full_J"] = nonadd([tuple(r["combo"]) for r in rows if r["cfg"] == "1111"])
    res["M_nonadditivity_generated_additive_J"] = nonadd([tuple(r["combo"]) for r in rows if r["cfg"] == "1110"])
    res["infeasible_init_total"] = int(sum(r["infeasible_init"] for r in raw))
    json.dump(res, open(os.path.join(OUT, "generation_analysis.json"), "w"), indent=2, default=float)
    figure1(res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("c_config_means",)}, indent=1, default=float)[:9000])


def figure1(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    eff = res["c_factorial_main_effects_on_output_metrics"]
    mets = [("T", "taste T"), ("C", "composition C"), ("N", "chem. novelty N"), ("M", "mixture M"), ("T_alt", "independent taste\n(descriptor GBM)"), ("frac_denovo", "frac. de novo")]
    cols = {"taste": "#1b6ca8", "composition": "#d98c00", "novelty": "#2a9d55", "mixture": "#b03a6e"}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.7, 1]})
    a = ax[0]; h = 0.18
    for j, (m, lab) in enumerate(mets):
        for i, t in enumerate(["taste", "composition", "novelty", "mixture"]):
            e = eff[m][t]; yv = j + (i - 1.5) * h; s = eff[m]["sd_across_runs"]
            a.errorbar(e["effect"] / s, yv, xerr=[[(e["effect"] - e["ci95"][0]) / s], [(e["ci95"][1] - e["effect"]) / s]], fmt="o", color=cols[t], ms=4, capsize=2, label=f"weight on {t}" if j == 0 else None)
    a.axvline(0, color="k", lw=0.8); a.set_yticks(range(len(mets))); a.set_yticklabels([l for _, l in mets]); a.invert_yaxis()
    a.set_xlabel("effect of switching a weight from 0 to its base value\n(in sd of the output across runs; 95% CI, OLS, 16 configs x 5 seeds)")
    a.legend(fontsize=7, loc="lower right", frameon=False); a.set_title("A. Which objective terms move which outputs", fontsize=10)
    b = ax[1]; d = res["c_drop_one_jaccard_vs_full"]; names = list(d)
    b.bar(range(4), [d[n]["jaccard_union_vs_full_same_seed"] for n in names], yerr=[[d[n]["jaccard_union_vs_full_same_seed"] - d[n]["ci95"][0] for n in names], [d[n]["ci95"][1] - d[n]["jaccard_union_vs_full_same_seed"] for n in names]], color=[cols[n] for n in names], capsize=3)
    b.axhline(res["c_noise_floor_jaccard"], color="k", ls="--", lw=1); b.text(3.4, res["c_noise_floor_jaccard"] + .01, "seed-to-seed\nnoise floor", fontsize=7, ha="right", va="bottom")
    b.axhline(res["c_random_feasible_baseline_0000_jaccard_vs_full"], color="gray", ls=":", lw=1)
    b.set_xticks(range(4)); b.set_xticklabels([f"drop\n{n}" for n in names], fontsize=8); b.set_ylabel("Jaccard overlap of molecules used\nvs full-J set (same seed)")
    b.set_title("B. Set change when one term is removed", fontsize=10)
    for x in ax: x.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); os.makedirs(FIG, exist_ok=True); plt.savefig(os.path.join(FIG, "fig1_weight_ablation.pdf")); plt.close()


if __name__ == "__main__":
    main()
