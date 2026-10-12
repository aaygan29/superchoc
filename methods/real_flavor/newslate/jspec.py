"""NEWSLATE: explicit, fast, config-driven implementation of the multi-objective search
J = w_taste*T + w_comp*C + w_novel*N + w_mix*M (see configs/J_spec.json and J_SPEC.md).

Reuses the legacy pool/model construction (denovo_recipes._prep) so that, with
taste_source='insample_legacy' and w_mix=0, it must reproduce the legacy generator exactly
(tested in test_jspec.py). The new term M scores a whole blend with the non-additive set model
trained on real human mixture-similarity data.
"""
import json, os, sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import denovo_recipes as dr
import composition as comp
import mixtures as mx
from reference_flavors import REFERENCE_FLAVORS
from featurize import smiles_to_ecfp4
import real_mixture_eval as rme

SPEC = json.load(open(os.path.join(HERE, "configs", "J_spec.json")))
REFS = list(REFERENCE_FLAVORS.keys())


def _tan_matrix(Xa, Xb):
    A = Xa.astype(np.float32); B = Xb.astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def stats_of(Z):
    return {"mean": Z.mean(0), "std": Z.std(0), "min": Z.min(0), "max": Z.max(0)}


def pair_feature(a, b):
    cos = float(a["mean"] @ b["mean"] / (np.linalg.norm(a["mean"]) * np.linalg.norm(b["mean"]) + 1e-12))
    return np.concatenate([[np.arccos(np.clip(cos, -1, 1))]] + [np.abs(a[k] - b[k]) for k in ("mean", "std", "min", "max")])


class Context:
    def __init__(self, taste_source="oof", mix_trees=300, seed=0):
        df, pool, X, model, stats, target, ref_fp = dr._prep()
        self.df, self.pool, self.X, self.model, self.target = df, pool, X, model, target
        n = len(pool); self.n = n
        self.elig = np.array([i for i, p in enumerate(pool) if p["safety"] == "OK"])
        self.is_known = np.array([p["source"] == "known" for p in pool])
        # --- T: per-molecule predicted pleasantness
        full = model.predict(X)
        oof = full.copy()
        kidx = np.where(self.is_known)[0]
        y = np.array([pool[i]["pleasantness"] for i in kidx])
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(kidx):
            m = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(X[kidx[tr]], y[tr])
            oof[kidx[te]] = m.predict(X[kidx[te]])
        self.taste_full, self.taste_oof = full, oof
        self.taste = oof if taste_source == "oof" else full
        self.taste_source = taste_source
        # --- C
        self.cval = np.array([comp.composition_score([p["smiles"]], target) for p in pool])
        # --- N
        self.S = np.zeros((n, len(REFS)))
        for f, name in enumerate(REFS):
            R = np.vstack([smiles_to_ecfp4(s) for s in REFERENCE_FLAVORS[name].values()])
            self.S[:, f] = _tan_matrix(X, R).max(1)
        # --- M: set model on real human mixture data
        snitz, ravia, cid2smi = rme.load_pairs()
        self.F = rme.Feat(cid2smi)
        Xs = np.array([self.F.pair(A, B, "set") for A, B, _ in snitz + ravia]); ys = np.array([s for *_, s in snitz + ravia])
        self.mix_model = RandomForestRegressor(n_estimators=mix_trees, min_samples_leaf=2, random_state=0, n_jobs=1).fit(Xs, ys)
        self.n_mix_train = len(ys)
        self.Z = []
        for p in pool:
            z = self.F.space._z(p["smiles"]); self.Z.append(z)
        bad = [i for i, z in enumerate(self.Z) if z is None]
        if bad: self.elig = np.array([i for i in self.elig if i not in set(bad)])
        self.Z = np.vstack([z if z is not None else np.zeros(13) for z in self.Z])
        self.ref_stats = []
        for name in REFS:
            Zr = np.vstack([self.F.space._z(s) for s in REFERENCE_FLAVORS[name].values() if self.F.space._z(s) is not None])
            self.ref_stats.append(stats_of(Zr))

    # ---- terms ----
    def N(self, combo):
        return float(1.0 - self.S[list(combo)].mean(0).max())

    def M(self, combo):
        a = stats_of(self.Z[list(combo)])
        Fm = np.vstack([pair_feature(a, r) for r in self.ref_stats])
        return float((100.0 - self.mix_model.predict(Fm).max()) / 100.0)

    def terms(self, combo, need_M=True):
        c = list(combo)
        return {"T": float(self.taste[c].mean() / 100.0), "C": float(self.cval[c].mean()),
                "N": self.N(combo), "M": self.M(combo) if need_M else 0.0}

    def J(self, combo, w):
        t = self.terms(combo, need_M=w["mixture"] != 0)
        return w["taste"] * t["T"] + w["composition"] * t["C"] + w["novelty"] * t["N"] + w["mixture"] * t["M"], t


def search(ctx, w, seed=0, n_restarts=5, iters=500, n_comp=10, nov_min=0.5, elig=None):
    rng = np.random.default_rng(seed); elig = ctx.elig if elig is None else elig
    recs, infeasible_init = [], 0
    for _ in range(n_restarts):
        combo = tuple(rng.choice(elig, size=n_comp, replace=False)); t = 0
        while ctx.N(combo) < nov_min and t < 100:
            combo = tuple(rng.choice(elig, size=n_comp, replace=False)); t += 1
        if ctx.N(combo) < nov_min: infeasible_init += 1
        bestJ = ctx.J(combo, w)[0]
        for _ in range(iters):
            v = int(rng.choice(elig)); k = int(rng.integers(n_comp))
            cand = list(combo); cand[k] = v
            if len(set(cand)) < n_comp: continue
            cand = tuple(cand)
            if ctx.N(cand) < nov_min: continue
            Jc = ctx.J(cand, w)[0]
            if Jc > bestJ: combo, bestJ = cand, Jc
        if frozenset(combo) not in {frozenset(c) for c, _ in recs}: recs.append((combo, bestJ))
    recs.sort(key=lambda r: r[1], reverse=True)
    out = []
    for combo, J in recs:
        J2, t = ctx.J(combo, {**w})
        t["M"] = ctx.M(combo)  # always report M, even when w_mix=0
        out.append({"combo": [int(i) for i in combo], "J": float(J), "terms": t})
    return out, infeasible_init
