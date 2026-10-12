"""NEWSLATE audit of (e) CV split type, (f) receptor absolute probabilities, (h) Lean statement,
(i) concentration dependence in the repo's own Keller data. Writes results/newslate/audit_claims.json.

Run from anywhere:  python3 methods/real_flavor/newslate/audit_claims.py
"""
import json, os, subprocess, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GroupKFold
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from real_data import build_keller_pleasantness
from featurize import smiles_to_ecfp4
import receptors as rc
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

OUT = os.path.abspath(os.path.join(HERE, "..", "results", "newslate"))
os.makedirs(OUT, exist_ok=True)


def boot_ci(y, p, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed); N = len(y); v = []
    for _ in range(n):
        i = rng.integers(0, N, N); v.append(fn(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def rho(a, b): return float(spearmanr(a, b).correlation)


def cv_preds(X, y, splits, seed=0):
    p = np.zeros(len(y))
    for tr, te in splits:
        m = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


def main():
    res = {}
    df = build_keller_pleasantness().reset_index(drop=True)
    X = np.vstack([smiles_to_ecfp4(s) for s in df.smiles]); y = df.pleasantness.values
    mols = [Chem.MolFromSmiles(s) for s in df.smiles]
    fps = [Chem.RDKFingerprint(m) if False else __import__("rdkit.Chem.rdFingerprintGenerator", fromlist=["x"]).GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(m) for m in mols]
    n = len(df)

    # ---- (e) splits ----
    # Butina clusters at Tanimoto distance 0.6 (sim >= 0.4), then GroupKFold
    dists = []
    for i in range(1, n):
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]); dists.extend([1 - x for x in s])
    cl = Butina.ClusterData(dists, n, 0.6, isDistData=True)
    grp_b = np.zeros(n, int)
    for k, c in enumerate(cl):
        for i in c: grp_b[i] = k
    # Murcko generic scaffolds (acyclics share the empty scaffold)
    scaf = []
    for m in mols:
        try:
            s = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(m)))
        except Exception:
            s = ""
        scaf.append(s)
    uniq = {s: i for i, s in enumerate(sorted(set(scaf)))}; grp_s = np.array([uniq[s] for s in scaf])
    res["split_info"] = {"n": n, "n_butina_clusters_sim>=0.4": int(grp_b.max() + 1),
                         "largest_butina_cluster": int(np.bincount(grp_b).max()),
                         "n_generic_scaffolds": len(uniq),
                         "frac_acyclic_empty_scaffold": float(np.mean([s == "" for s in scaf]))}
    schemes = {}
    seeds = [0, 1, 2]
    for name in tqdm(["random_kfold", "butina_groupkfold", "scaffold_groupkfold"], desc="splits"):
        rs, r2s, allp = [], [], []
        for sd in seeds:
            if name == "random_kfold":
                sp = list(KFold(5, shuffle=True, random_state=sd).split(X))
            else:
                g = grp_b if name.startswith("butina") else grp_s
                rng = np.random.default_rng(sd); perm = rng.permutation(g.max() + 1); g2 = perm[g]  # shuffle group order
                sp = list(GroupKFold(5).split(X, y, g2))
            p = cv_preds(X, y, sp, seed=sd); allp.append(p)
            rs.append(rho(y, p)); r2s.append(float(r2_score(y, p)))
            # max-train-similarity of each test molecule
        # mean max Tanimoto to train for test molecules (seed 0 splits)
        if name == "random_kfold":
            sp0 = list(KFold(5, shuffle=True, random_state=0).split(X))
        else:
            g = grp_b if name.startswith("butina") else grp_s
            rng = np.random.default_rng(0); perm = rng.permutation(g.max() + 1)
            sp0 = list(GroupKFold(5).split(X, y, perm[g]))
        mx = np.zeros(n)
        for tr, te in sp0:
            for i in te: mx[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in tr]))
        pm = np.mean(allp, 0)
        schemes[name] = {"spearman_per_seed": rs, "spearman_mean": float(np.mean(rs)),
                         "spearman_ci95_seed0_boot": boot_ci(y, allp[0], rho),
                         "r2_per_seed": r2s, "r2_mean": float(np.mean(r2s)),
                         "mean_max_tanimoto_test_to_train": float(mx.mean()),
                         "median_max_tanimoto_test_to_train": float(np.median(mx))}
        if name == "random_kfold": mx_random = mx
        if name == "butina_groupkfold": mx_but = mx; pbut = allp[0]
    res["pleasantness_cv"] = schemes
    res["pleasantness_cv"]["repo_code_split_type"] = ("goodness_real.cross_validate uses sklearn KFold(shuffle=True): "
                                                     "RANDOM over molecules, no similarity awareness")
    # stratify random-CV performance by max train similarity
    pr = cv_preds(X, y, list(KFold(5, shuffle=True, random_state=0).split(X)), seed=0)
    bins = [(0, .3), (.3, .4), (.4, .5), (.5, .7), (.7, 1.01)]
    strat = []
    for lo, hi in bins:
        k = (mx_random >= lo) & (mx_random < hi)
        strat.append({"max_sim_bin": [lo, hi], "n": int(k.sum()),
                      "spearman": rho(y[k], pr[k]) if k.sum() > 5 else None})
    res["random_cv_spearman_by_max_train_similarity"] = strat

    # ---- (f) receptor absolute probabilities ----
    pairs = rc.load_mainland()
    genes = [p["or_gene"] for p in pairs]; cids = [p["cid"] for p in pairs]; N = len(pairs)
    from collections import Counter
    cnt = Counter(genes)
    base = rc.validate_leave_one_out(pairs)
    # leave-MOLECULE-out: nearest neighbour must be a different molecule (CID)
    fpm = [p["fp"] for p in pairs]
    hits = 0; same_mol_nn = 0; nnsim = []
    for i in range(N):
        sims = [(rc._tanimoto(fpm[i], fpm[j]), j) for j in range(N) if j != i]
        s, j = max(sims)
        same_mol_nn += int(cids[j] == cids[i])
        hits += int(genes[i] == genes[j])
    hits_lmo = 0; nlmo = 0; nn_s = []
    for i in range(N):
        sims = [(rc._tanimoto(fpm[i], fpm[j]), j) for j in range(N) if cids[j] != cids[i]]
        if not sims: continue
        s, j = max(sims); nn_s.append(s); hits_lmo += int(genes[i] == genes[j]); nlmo += 1
    chance_orig = sum((v / N) ** 2 for v in cnt.values())
    # chance for the lmo design: P(gene_i == gene_j) over random j with cid_j != cid_i
    ch = []
    for i in range(N):
        js = [j for j in range(N) if cids[j] != cids[i]]
        ch.append(np.mean([genes[j] == genes[i] for j in js]))
    chance_lmo = float(np.mean(ch))
    # bootstrap CI over query ligands for original rate
    hit_vec = []
    for i in range(N):
        sims = [(rc._tanimoto(fpm[i], fpm[j]), j) for j in range(N) if j != i]; _, j = max(sims)
        hit_vec.append(int(genes[i] == genes[j]))
    hit_vec = np.array(hit_vec); rng = np.random.default_rng(0)
    bs = [hit_vec[rng.integers(0, N, N)].mean() for _ in range(2000)]
    # per-molecule gene lists: how many pairs per molecule
    res["receptor_claim"] = {
        "n_pairs": N, "n_molecules": len(set(cids)), "n_distinct_genes": len(cnt),
        "repo_validate_leave_one_out": base,
        "repo_nn_shared_rate_absolute": base["nn_shared_or_rate"], "repo_chance_absolute": base["chance_rate"],
        "repo_lift": base["lift"],
        "nn_hit_rate_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "frac_of_queries_whose_NN_is_the_same_molecule_in_another_row": same_mol_nn / N,
        "leave_molecule_out_nn_shared_rate": hits_lmo / nlmo, "leave_molecule_out_chance": chance_lmo,
        "leave_molecule_out_lift": (hits_lmo / nlmo) / chance_lmo if chance_lmo else None,
        "leave_molecule_out_mean_nn_tanimoto": float(np.mean(nn_s)),
        "top_gene_freq": cnt.most_common(5),
    }

    # ---- (h) Lean ----
    lean_path = os.path.join(os.path.dirname(HERE), "FlavorMath.lean")
    try:
        r = subprocess.run(["lean", lean_path], capture_output=True, text=True, timeout=120,
                           env={**os.environ, "PATH": os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"]})
        lean = {"exit_code": r.returncode, "stdout": r.stdout[:500], "stderr": r.stderr[:500]}
    except Exception as e:
        lean = {"error": repr(e)}
    src = open(lean_path).read()
    lean["theorem_statements"] = [l.strip() for l in src.splitlines() if l.strip().startswith("theorem")]
    lean["mentions_weights_or_convex"] = any(w in src.split("-/", 1)[1] for w in ["Rat", "Real", "∑", "foldl", "weights : "])
    lean["reading"] = ("The checked theorems state only that every element of a list of Nat is <= the list max "
                       "(listMax is defined by iterated max). That is the definition of max. Nothing in the file "
                       "formalizes weights, convexity, or the real-valued tastiness T(R)=sum w_i p_i; the "
                       "'convex combination <= max' step is prose. It is a definitional fact for a footnote, not a result.")
    res["lean"] = lean

    # ---- (i) concentration dependence in Keller 2016 (measured) ----
    import pyrfume, pandas as pd
    beh = pyrfume.load_data("keller_2016/behavior.csv"); stim = pyrfume.load_data("keller_2016/stimuli.csv").reset_index()
    mv = beh["MeasurementValue"].astype(str).str.upper()
    pl = beh[mv.str.contains("PLEASANT", na=False)].copy(); pl["num"] = pd.to_numeric(pl["Value"], errors="coerce")
    sp = pl.dropna(subset=["num"]).groupby("Stimulus")["num"].mean().rename("pl").reset_index()
    s = stim.copy(); s["CIDs"] = s["CIDs"].astype(str)
    s = s[~s["CIDs"].str.contains(r"[;, ]", regex=True, na=False)].merge(sp, on="Stimulus")
    piv = s.pivot_table(index="CIDs", columns="Ratio", values="pl", aggfunc="mean")
    conc = {}
    for a, b in [("1/1000", "1/10"), ("1/1000", "1/100000")]:
        d = piv[[a, b]].dropna()
        diff = (d[b] - d[a])
        conc[f"{a}_vs_{b}"] = {"n_molecules": int(len(d)), "spearman": rho(d[a], d[b]),
                               "mean_abs_change_pts": float(diff.abs().mean()),
                               "frac_abs_change_ge_10pts": float((diff.abs() >= 10).mean()),
                               "frac_sign_flip_vs_mean45.9": float((((d[a] - 45.9) * (d[b] - 45.9)) < 0).mean()),
                               "sd_of_pleasantness_at_1/1000": float(d[a].std())}
    res["concentration_keller2016"] = conc
    json.dump(res, open(os.path.join(OUT, "audit_claims.json"), "w"), indent=2, default=float)
    print(json.dumps({k: res[k] for k in ["split_info", "random_cv_spearman_by_max_train_similarity"]}, indent=1, default=float))
    for k, v in schemes.items(): print(k, round(v["spearman_mean"], 3), round(v["r2_mean"], 3), round(v["mean_max_tanimoto_test_to_train"], 3))
    print(json.dumps(res["receptor_claim"], indent=1, default=float)); print(lean); print(json.dumps(conc, indent=1))


if __name__ == "__main__":
    main()
