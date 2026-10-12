"""NEWSLATE (d): held-out evaluation of the non-additive (set-pooled) mixture model on REAL human
mixture-similarity data (Snitz 2013, n=360 pairs; Ravia 2020, n=195 pairs), beside the synthetic
oracle number. Writes results/newslate/real_mixture_eval.json and a cached pair table.

Models (all predict human similarity of a mixture PAIR from component descriptors, 13 RDKit
physchem descriptors z-scored over all dataset molecules; labels never touch the z-scoring):
  angle_untrained : Snitz/Ravia additive angle between mean vectors (no fitting; sign flipped)
  meanpool_RF     : RF on [angle, |d mean|]                      (additive representation, learned)
  set_RF          : RF on [angle, |d mean|, |d std|, |d min|, |d max|]  (non-additive pooling, learned)
  size_only_RF    : RF on [|log nA - log nB|, log nA + log nB]   (control: component-count artifact)
  set_RF_permuted : set_RF trained on shuffled labels            (control: chance)
Protocols: Snitz->Ravia, Ravia->Snitz (disjoint mixtures AND disjoint labs), and within-dataset
leave-mixture-out 5-fold (train pairs touch no held-out mixture; test pairs touch >=1).
"""
import json, os, re, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mixtures as mx
from rdkit import Chem

OUT = os.path.abspath(os.path.join(HERE, "..", "results", "newslate"))
os.makedirs(OUT, exist_ok=True)


def load_pairs():
    import pyrfume
    sm = pyrfume.load_data("snitz_2013/molecules.csv")
    cs = {int(c): r["IsomericSMILES"] for c, r in sm.iterrows() if isinstance(r["IsomericSMILES"], str)}
    b = pyrfume.load_data("snitz_2013/behavior.csv")
    snitz = []
    for _, r in b.iterrows():
        A = tuple(sorted(int(t) for t in str(r["StimulusA"]).split(",") if t.strip().isdigit() and int(t) in cs))
        B = tuple(sorted(int(t) for t in str(r["StimulusB"]).split(",") if t.strip().isdigit() and int(t) in cs))
        if A and B: snitz.append((A, B, float(r["Similarity"])))
    st = pyrfume.load_data("ravia_2020/stimuli.csv")
    rm = pyrfume.load_data("ravia_2020/molecules.csv")
    cr = {int(c): r["IsomericSMILES"] for c, r in rm.iterrows() if isinstance(r["IsomericSMILES"], str)}
    b2 = pyrfume.load_data("ravia_2020/behavior_2.csv").reset_index()

    def stim(sid):
        if sid not in st.index: return ()
        return tuple(sorted(int(t) for t in re.split(r"[;,]", str(st.loc[sid, "CID"]))
                            if t.strip().isdigit() and int(t) in cr))
    ravia = []
    for _, r in b2.iterrows():
        A, B = stim(r["Stimulus 1"]), stim(r["Stimulus 2"])
        if A and B: ravia.append((A, B, float(r["RatedSimilarity"])))
    cid2smi = {**cs, **cr}
    return snitz, ravia, cid2smi


class Feat:
    def __init__(self, cid2smi):
        self.cid2smi = {c: s for c, s in cid2smi.items() if Chem.MolFromSmiles(s) is not None}
        self.space = mx.MixtureSpace(list(self.cid2smi.values()))
        self.cache = {}

    def mix(self, cids):
        if cids in self.cache: return self.cache[cids]
        Z = np.vstack([self.space._z(self.cid2smi[c]) for c in cids if c in self.cid2smi
                       and self.space._z(self.cid2smi[c]) is not None])
        v = {"mean": Z.mean(0), "std": Z.std(0), "min": Z.min(0), "max": Z.max(0), "n": len(Z)}
        self.cache[cids] = v; return v

    def pair(self, A, B, kind):
        a, b = self.mix(A), self.mix(B)
        cos = float(a["mean"] @ b["mean"] / (np.linalg.norm(a["mean"]) * np.linalg.norm(b["mean"]) + 1e-12))
        ang = float(np.arccos(np.clip(cos, -1, 1)))
        if kind == "angle": return [ang]
        if kind == "meanpool": return [ang] + list(np.abs(a["mean"] - b["mean"]))
        if kind == "set":
            return [ang] + [x for k in ("mean", "std", "min", "max") for x in np.abs(a[k] - b[k])]
        if kind == "size":
            la, lb = np.log(a["n"]), np.log(b["n"]); return [abs(la - lb), la + lb]


def design(F, pairs, kind):
    return np.array([F.pair(A, B, kind) for A, B, _ in pairs]), np.array([s for _, _, s in pairs])


def rf(seed): return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1)


def rho(a, b): return float(spearmanr(a, b).correlation)


def boot(y, preds: dict, n=2000, seed=0):
    """Bootstrap pairs -> CI for each model's Spearman and for set - meanpool difference."""
    rng = np.random.default_rng(seed); N = len(y); R = {k: [] for k in preds}; D = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        r = {k: rho(y[i], p[i]) for k, p in preds.items()}
        for k in r: R[k].append(r[k])
        D.append(r["set_RF"] - r["meanpool_RF"])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {k: ci(v) for k, v in R.items()}, ci(D)


def run_protocol(F, train, test, seeds, name):
    Xtr = {k: design(F, train, k)[0] for k in ("meanpool", "set", "size")}; ytr = np.array([s for *_, s in train])
    Xte = {k: design(F, test, k)[0] for k in ("angle", "meanpool", "set", "size")}; yte = np.array([s for *_, s in test])
    P = {"angle_untrained": -Xte["angle"][:, 0]}
    acc = {"meanpool_RF": [], "set_RF": [], "size_only_RF": [], "set_RF_permuted": []}
    for sd in seeds:
        acc["meanpool_RF"].append(rf(sd).fit(Xtr["meanpool"], ytr).predict(Xte["meanpool"]))
        acc["set_RF"].append(rf(sd).fit(Xtr["set"], ytr).predict(Xte["set"]))
        acc["size_only_RF"].append(rf(sd).fit(Xtr["size"], ytr).predict(Xte["size"]))
        acc["set_RF_permuted"].append(rf(sd).fit(Xtr["set"], np.random.default_rng(sd).permutation(ytr)).predict(Xte["set"]))
    for k, v in acc.items(): P[k] = np.mean(v, 0)
    per_seed = {k: [rho(yte, p) for p in v] for k, v in acc.items()}
    cis, dci = boot(yte, {"set_RF": P["set_RF"], "meanpool_RF": P["meanpool_RF"],
                          "angle_untrained": P["angle_untrained"], "size_only_RF": P["size_only_RF"],
                          "set_RF_permuted": P["set_RF_permuted"]})
    out = {"n_train_pairs": len(train), "n_test_pairs": len(test),
           "spearman": {k: rho(yte, p) for k, p in P.items()},
           "spearman_ci95_pair_bootstrap": cis,
           "spearman_per_rf_seed": per_seed,
           "set_minus_meanpool": rho(yte, P["set_RF"]) - rho(yte, P["meanpool_RF"]),
           "set_minus_meanpool_ci95": dci}
    return out, P, yte


def lmo_folds(pairs, k=5, seed=0):
    mixes = sorted({m for A, B, _ in pairs for m in (A, B)})
    rng = np.random.default_rng(seed); perm = rng.permutation(len(mixes)); fold = {mixes[i]: j % k for j, i in enumerate(perm)}
    for f in range(k):
        te = [p for p in pairs if fold[p[0]] == f or fold[p[1]] == f]
        tr = [p for p in pairs if fold[p[0]] != f and fold[p[1]] != f]
        yield tr, te


def within(F, pairs, seeds, name):
    """Leave-mixture-out CV. Test pairs pooled over folds; each pair may appear in 2 folds (both
    mixtures held out in different folds), so we average its predictions across appearances."""
    N = len(pairs); idx = {id(p): i for i, p in enumerate(pairs)}
    S = {k: np.zeros(N) for k in ("meanpool_RF", "set_RF", "size_only_RF", "set_RF_permuted")}; C = np.zeros(N)
    ang = np.zeros(N); ntrain = []
    for tr, te in tqdm(list(lmo_folds(pairs)), desc=f"LMO {name}"):
        if len(tr) < 20: continue
        ntrain.append(len(tr))
        Xtr = {k: design(F, tr, k)[0] for k in ("meanpool", "set", "size")}; ytr = np.array([s for *_, s in tr])
        Xte = {k: design(F, te, k)[0] for k in ("angle", "meanpool", "set", "size")}
        ti = [idx[id(p)] for p in te]
        for sd in seeds:
            S["meanpool_RF"][ti] += rf(sd).fit(Xtr["meanpool"], ytr).predict(Xte["meanpool"])
            S["set_RF"][ti] += rf(sd).fit(Xtr["set"], ytr).predict(Xte["set"])
            S["size_only_RF"][ti] += rf(sd).fit(Xtr["size"], ytr).predict(Xte["size"])
            S["set_RF_permuted"][ti] += rf(sd).fit(Xtr["set"], np.random.default_rng(sd).permutation(ytr)).predict(Xte["set"])
        C[ti] += len(seeds); ang[ti] = -Xte["angle"][:, 0]
    m = C > 0; y = np.array([s for *_, s in pairs])[m]
    P = {k: v[m] / C[m] for k, v in S.items()}; P["angle_untrained"] = ang[m]
    cis, dci = boot(y, {k: P[k] for k in ("set_RF", "meanpool_RF", "angle_untrained", "size_only_RF", "set_RF_permuted")})
    return {"n_test_pairs_scored": int(m.sum()), "mean_train_pairs_per_fold": float(np.mean(ntrain)),
            "spearman": {k: rho(y, p) for k, p in P.items()}, "spearman_ci95_pair_bootstrap": cis,
            "set_minus_meanpool": rho(y, P["set_RF"]) - rho(y, P["meanpool_RF"]), "set_minus_meanpool_ci95": dci}


def synthetic_oracle(seeds=range(5)):
    import mixture_model as mm
    res = {}
    for supp in (0.8, 0.0):
        rows = []
        for sd in seeds:
            o = mm.NonAdditiveMixtureOracle(suppression=supp, seed=sd)
            rng = np.random.default_rng(sd + 1)
            combos = mm._make_combos(o.pool.shape[0], 4, 1500, rng)
            y = np.array([o.query_mixture(c) for c in combos])
            from sklearn.model_selection import train_test_split
            r = {}
            for name, feat in [("mean_pool", mm.mean_pool_features), ("set", mm.set_features)]:
                X = feat(o, combos); a, b, c, d = train_test_split(X, y, test_size=0.3, random_state=sd)
                r[name] = rho(d, RandomForestRegressor(200, random_state=sd, n_jobs=-1).fit(a, c).predict(b))
            rows.append(r)
        res[f"suppression_{supp}"] = {"mean_pool_mean": float(np.mean([r["mean_pool"] for r in rows])),
                                      "set_mean": float(np.mean([r["set"] for r in rows])),
                                      "mean_pool_sd": float(np.std([r["mean_pool"] for r in rows])),
                                      "set_sd": float(np.std([r["set"] for r in rows])), "per_seed": rows}
    return res


def main():
    snitz, ravia, cid2smi = load_pairs()
    F = Feat(cid2smi)
    seeds = [0, 1, 2, 3, 4]
    out = {"n_snitz_pairs": len(snitz), "n_ravia_pairs": len(ravia),
           "n_snitz_mixtures": len({m for A, B, _ in snitz for m in (A, B)}),
           "n_ravia_mixtures": len({m for A, B, _ in ravia for m in (A, B)}),
           "snitz_singleton_mixtures": len({m for A, B, _ in snitz for m in (A, B) if len(m) == 1}),
           "snitz_pairs_that_are_both_singletons": sum(len(A) == 1 and len(B) == 1 for A, B, _ in snitz),
           "ravia_pairs_that_are_both_singletons": sum(len(A) == 1 and len(B) == 1 for A, B, _ in ravia)}
    out["cross_dataset"] = {}
    r, _, _ = run_protocol(F, snitz, ravia, seeds, "snitz->ravia"); out["cross_dataset"]["train_Snitz_test_Ravia"] = r
    r, _, _ = run_protocol(F, ravia, snitz, seeds, "ravia->snitz"); out["cross_dataset"]["train_Ravia_test_Snitz"] = r
    out["within_dataset_leave_mixture_out"] = {"Snitz": within(F, snitz, seeds[:3], "snitz"),
                                               "Ravia": within(F, ravia, seeds[:3], "ravia")}
    out["synthetic_oracle"] = synthetic_oracle()
    json.dump(out, open(os.path.join(OUT, "real_mixture_eval.json"), "w"), indent=2, default=float)
    for k, v in out["cross_dataset"].items(): print(k, {a: round(b, 3) for a, b in v["spearman"].items()}, round(v["set_minus_meanpool"], 3), v["set_minus_meanpool_ci95"])
    for k, v in out["within_dataset_leave_mixture_out"].items(): print("LMO", k, v["n_test_pairs_scored"], {a: round(b, 3) for a, b in v["spearman"].items()}, round(v["set_minus_meanpool"], 3), v["set_minus_meanpool_ci95"])
    print(json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items() if kk != "per_seed"} for k, v in out["synthetic_oracle"].items()}, indent=1))


if __name__ == "__main__":
    main()
