"""NEWSLATE (g): synthetic accessibility + alternative-generator baselines for the de novo molecules.

Guided pipeline set  = top-40 of results/lego_denovo.json by predicted pleasantness (what J draws on).
Baselines (same safety + novel-vs-Keller filters, same predictor, same criteria):
  U1 unguided BRICS, same enumeration: 40 random draws from the same filtered candidate list (x300)
  U2 unguided BRICS, re-seeded: random 60-fragment subsets of the fragment library, BRICSBuild,
     same filters, 40 random draws (x8 seeds)
  K  known safe Keller molecules (no generator), 40 random draws (x300): reference for SA / pleasantness
Criteria (declared before running): predicted pleasantness >= Keller 75th percentile; max ECFP4
Tanimoto to any Keller molecule < 0.5; SA score <= 4.5 (Ertl & Schuffenhauer 2009; ~>6 is hard).
Because the guided set is selected on the predictor, we also score every set with an independent
descriptor-based GBM trained on the same labels (selection-bias audit).
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import Descriptors, inchi, rdFingerprintGenerator, BRICS
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from real_data import build_keller_pleasantness
from safety import screen
import lego_assembly as la

OUT = os.path.abspath(os.path.join(HERE, "..", "results", "newslate"))
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(s): return gen.GetFingerprint(Chem.MolFromSmiles(s))
def desc(s):
    d = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)); return np.array(list(d.values()), float)


def main():
    rng = np.random.default_rng(0)
    df = build_keller_pleasantness().reset_index(drop=True)
    kfps = [fp(s) for s in df.smiles]; y = df.pleasantness.values
    Xe = np.array([list(f) for f in kfps], dtype=np.int8)
    modelA = RandomForestRegressor(300, random_state=0, n_jobs=-1).fit(Xe, y)
    D = np.nan_to_num(np.vstack([desc(s) for s in df.smiles]), posinf=0, neginf=0)
    D = np.clip(D, -1e6, 1e6)
    modelB = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(D, y)
    # quality of model B (random 5-fold) for the record
    pb = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(D):
        pb[te] = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(D[tr], y[tr]).predict(D[te])
    q75 = float(np.quantile(y, 0.75))
    known_ik = {inchi.MolToInchiKey(Chem.MolFromSmiles(s)) for s in df.smiles}
    safe_known = [s for s in df.smiles if screen(s)["verdict"] == "OK"]

    def metrics(smis):
        smis = list(smis)
        Xs = np.array([list(fp(s)) for s in smis], dtype=np.int8)
        pa = modelA.predict(Xs)
        Ds = np.clip(np.nan_to_num(np.vstack([desc(s) for s in smis]), posinf=0, neginf=0), -1e6, 1e6)
        pb_ = modelB.predict(Ds)
        mt = np.array([max(DataStructs.BulkTanimotoSimilarity(fp(s), kfps)) for s in smis])
        sa = np.array([sascorer.calculateScore(Chem.MolFromSmiles(s)) for s in smis])
        ok = (pa >= q75) & (mt < 0.5) & (sa <= 4.5)
        return {"n": len(smis), "pred_pleasantness_A": float(pa.mean()), "pred_pleasantness_B_indep": float(pb_.mean()),
                "mean_max_tanimoto_to_keller": float(mt.mean()), "mean_SA": float(sa.mean()), "median_SA": float(np.median(sa)),
                "frac_SA_gt_6": float((sa > 6).mean()), "frac_SA_le_4.5": float((sa <= 4.5).mean()),
                "frac_joint_pass": float(ok.mean()),
                "frac_pass_pleasant_A": float((pa >= q75).mean()), "frac_pass_novel": float((mt < 0.5).mean())}, pa, pb_, mt, sa

    lego = json.load(open(os.path.join(os.path.dirname(HERE), "results", "lego_denovo.json")))
    cands = [c["smiles"] for c in lego["candidates"]]
    # re-score all with the same predictor for fairness (json used the in-script model, same seed)
    mAll, pa_all, pb_all, mt_all, sa_all = metrics(cands)
    order = np.argsort(-pa_all)
    guided = [cands[i] for i in order[:40]]
    res = {"q75_keller_pleasantness": q75, "n_candidates_in_lego_json": len(cands),
           "modelB_random_cv_spearman": float(spearmanr(y, pb).correlation),
           "all_434_candidates": mAll,
           "predA_vs_predB_spearman_on_candidates": float(spearmanr(pa_all, pb_all).correlation)}
    res["guided_top40_by_predA"], *_ = metrics(guided)
    # U1: random draws from same filtered list
    rows = []
    for _ in tqdm(range(300), desc="U1"):
        idx = rng.choice(len(cands), 40, replace=False); rows.append(metrics_fast := None)
    # (fast path: reuse precomputed per-candidate arrays)
    ok_all = (pa_all >= q75) & (mt_all < 0.5) & (sa_all <= 4.5)
    def agg(idxs):
        return {"pred_pleasantness_A": pa_all[idxs].mean(), "pred_pleasantness_B_indep": pb_all[idxs].mean(),
                "mean_max_tanimoto_to_keller": mt_all[idxs].mean(), "mean_SA": sa_all[idxs].mean(),
                "frac_SA_gt_6": (sa_all[idxs] > 6).mean(), "frac_joint_pass": ok_all[idxs].mean()}
    draws = [agg(rng.choice(len(cands), 40, replace=False)) for _ in range(300)]
    res["U1_unguided_same_enumeration_x300"] = {k: {"mean": float(np.mean([d[k] for d in draws])), "sd": float(np.std([d[k] for d in draws]))} for k in draws[0]}
    res["guided_vs_U1_selection_effect"] = {"predA_gain": float(pa_all[order[:40]].mean() - np.mean([d["pred_pleasantness_A"] for d in draws])),
                                            "predB_gain_(independent_model)": float(pb_all[order[:40]].mean() - np.mean([d["pred_pleasantness_B_indep"] for d in draws])),
                                            "SA_change": float(sa_all[order[:40]].mean() - np.mean([d["mean_SA"] for d in draws]))}
    # U2: re-seeded random-fragment BRICS
    safe_frags = la.fragment_library(safe_known)
    u2 = []
    for sd in tqdm(range(8), desc="U2 BRICS re-seeded"):
        r2 = np.random.default_rng(100 + sd)
        sub = [safe_frags[i] for i in r2.choice(len(safe_frags), 60, replace=False)]
        built = la.assemble(sub, n_build=400, seed=sd)
        pool = [b["smiles"] for b in built if b["inchikey"] not in known_ik and screen(b["smiles"])["verdict"] == "OK"]
        if len(pool) < 40: u2.append({"seed": sd, "n_pool": len(pool)}); continue
        pick = [pool[i] for i in r2.choice(len(pool), 40, replace=False)]
        m, *_ = metrics(pick); m["seed"] = sd; m["n_pool"] = len(pool); u2.append(m)
    res["U2_unguided_brics_reseeded"] = u2
    good = [r for r in u2 if "mean_SA" in r]
    res["U2_mean"] = {k: float(np.mean([r[k] for r in good])) for k in good[0] if k not in ("seed", "n")} if good else None
    # K
    kn = [metrics([safe_known[i] for i in rng.choice(len(safe_known), 40, replace=False)])[0] for _ in tqdm(range(50), desc="K")]
    res["K_known_safe_keller_x50"] = {k: float(np.mean([d[k] for d in kn])) for k in kn[0] if k != "n"}
    # PubChem absence vs SA among those with PubChem labels
    pc = [(c["smiles"], c.get("in_pubchem")) for c in lego["candidates"] if c.get("in_pubchem") is not None]
    res["pubchem_labelled_candidates"] = {"n": len(pc), "n_absent": sum(1 for _, k in pc if k is False),
                                          "mean_SA_of_pubchem_absent": float(np.mean([sascorer.calculateScore(Chem.MolFromSmiles(s)) for s, k in pc if k is False])) if any(k is False for _, k in pc) else None}
    json.dump(res, open(os.path.join(OUT, "generator_baselines.json"), "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("U2_unguided_brics_reseeded",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
