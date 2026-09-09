"""Flavor-space geometry: the mathematical structure of flavor space, and a layer that
locates a blend within it and hypothesizes where it should land.

Mathematical background (see docs / paper):
  - Flavor/odor space is a METRIC space: distance = perceptual dissimilarity
    (Kurtz et al. 2000, doi:10.3758/bf03212093).
  - It is empirically LOW-DIMENSIONAL despite huge molecular dimensionality, and is best
    fit by HYPERBOLIC (negative-curvature) geometry, because odor co-occurrence in natural
    mixtures is hierarchical / tree-like (Zhou, Smith & Sharpee 2018, Sci Adv,
    doi:10.1126/sciadv.aaq1458 -- a 3D hyperbolic space fits both natural odor statistics
    and human descriptions).
  - Learned Euclidean embeddings (Principal Odor Map) also make distance predict perceptual
    similarity (Lee et al. 2023).

What this module does, on our real data (Leffingwell odor descriptors as the perceptual
representation):
  1. tree-likeness / hyperbolicity test: normalized Gromov 4-point delta of the perceptual
     distance matrix vs a shuffled control (is flavor space tree-like, as Sharpee found?).
  2. a low-dimensional coordinate embedding (PCA of the descriptor space).
  3. locate(blend): place a blend as a point, report its coordinates, nearest known flavors,
     and a NOVELTY hypothesis = local density percentile (does it sit in a sparse, unexplored
     region of flavor space?).

This is an evaluation/hypothesis layer, not a perception ground truth: coordinates are a
model of where a blend lands, to be checked against a panel.
"""

from __future__ import annotations

import numpy as np

import flavor_profile as fp
from featurize import smiles_to_ecfp4


def _jaccard_dist(A):
    """Pairwise Jaccard distance for a binary matrix A (n x d)."""
    A = (A > 0).astype(float)
    inter = A @ A.T
    rs = A.sum(1)
    union = rs[:, None] + rs[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return 1.0 - sim


def gromov_delta(D, n_quads=20000, seed=0):
    """Normalized Gromov 4-point hyperbolicity delta of a distance matrix D.
    delta in [0, 1]; smaller = more tree-like (0 = a tree, hyperbolic). Sampled over
    random 4-tuples. Returns mean normalized delta."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    diam = D.max()
    if diam == 0:
        return 0.0
    deltas = []
    for _ in range(n_quads):
        i, j, k, l = rng.choice(n, size=4, replace=False)
        d_ij_kl = D[i, j] + D[k, l]
        d_ik_jl = D[i, k] + D[j, l]
        d_il_jk = D[i, l] + D[j, k]
        s = sorted([d_ij_kl, d_ik_jl, d_il_jk])
        deltas.append((s[2] - s[1]) / 2.0)
    return float(np.mean(deltas) / diam)


class FlavorSpaceGeometry:
    def __init__(self, dim=3, max_mols=2500, seed=0):
        self.dim = dim
        names, cid2desc = fp.load_descriptor_table()
        import pyrfume
        lm = pyrfume.load_data("leffingwell/molecules.csv")
        rows, vecs, fps = [], [], []
        for cid, r in lm.iterrows():
            if int(cid) not in cid2desc:
                continue
            try:
                fps.append(smiles_to_ecfp4(r["IsomericSMILES"]))
            except Exception:
                continue
            vecs.append(cid2desc[int(cid)])
            rows.append(r["name"])
            if len(rows) >= max_mols:
                break
        self.names = rows
        self.V = np.vstack(vecs)          # binary descriptor vectors
        self.FP = np.vstack(fps)          # ECFP for out-of-sample structural NN
        self.desc_names = names
        # low-dim coordinates via PCA of descriptor space (centered)
        Vc = self.V - self.V.mean(0)
        U, S, Wt = np.linalg.svd(Vc, full_matrices=False)
        self.components = Wt[:dim]
        self.coords = Vc @ self.components.T
        self._mean = self.V.mean(0)

    def hyperbolicity(self, n_sample=300, seed=0):
        """Compare tree-likeness of the perceptual distance matrix vs a shuffled control."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.names), size=min(n_sample, len(self.names)), replace=False)
        D = _jaccard_dist(self.V[idx])
        real = gromov_delta(D, seed=seed)
        # control: shuffle each feature column to destroy co-occurrence structure
        Vs = self.V[idx].copy()
        for c in range(Vs.shape[1]):
            rng.shuffle(Vs[:, c])
        Dctrl = _jaccard_dist(Vs)
        ctrl = gromov_delta(Dctrl, seed=seed)
        return {"gromov_delta_real": round(real, 4), "gromov_delta_shuffled": round(ctrl, 4),
                "more_tree_like_than_control": bool(real < ctrl)}

    def embedding_faithfulness(self, n_sample=300, seed=0):
        """Do low-dim EUCLIDEAN (PCA) coordinates preserve perceptual distances?"""
        from scipy.stats import spearmanr
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.names), size=min(n_sample, len(self.names)), replace=False)
        Dtrue = _jaccard_dist(self.V[idx])
        C = self.coords[idx]
        Demb = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(idx), 1)
        return round(float(spearmanr(Dtrue[iu], Demb[iu]).correlation), 4)

    def poincare_faithfulness(self, n_sample=250, dim=3, steps=800, seed=0):
        """Fit a Poincare-ball (hyperbolic) embedding to the perceptual distance matrix and
        compare its distance-preservation to the flat Euclidean (PCA) embedding at the same
        dimension. Sharpee et al. (2018) predict hyperbolic geometry fits odor space better;
        this tests that on our data. Returns both Spearman values and the winner.

        Poincare distance: d(u,v) = arcosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))."""
        import torch
        from scipy.stats import spearmanr

        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.names), size=min(n_sample, len(self.names)), replace=False)
        D = _jaccard_dist(self.V[idx])
        iu = np.triu_indices(len(idx), 1)
        target = torch.tensor(D[iu], dtype=torch.float64)

        torch.manual_seed(seed)
        n = len(idx)
        X = torch.nn.Parameter(torch.randn(n, dim, dtype=torch.float64) * 0.01)
        log_s = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))  # learnable scale
        opt = torch.optim.Adam([X, log_s], lr=5e-2)
        I, J = iu
        I = torch.tensor(I); J = torch.tensor(J)
        eps = 1e-6
        for _ in range(steps):
            opt.zero_grad()
            norm = (X * X).sum(1)
            norm = torch.clamp(norm, max=1 - 1e-4)
            u, v = X[I], X[J]
            nu, nv = norm[I], norm[J]
            diff2 = ((u - v) ** 2).sum(1)
            arg = 1 + 2 * diff2 / ((1 - nu) * (1 - nv) + eps)
            dP = torch.acosh(torch.clamp(arg, min=1 + eps))
            loss = ((dP - torch.exp(log_s) * target) ** 2).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():                     # keep points inside the ball
                nrm = X.norm(dim=1, keepdim=True)
                too_big = (nrm > 1 - 1e-3).squeeze()
                if too_big.any():
                    X[too_big] = X[too_big] / nrm[too_big] * (1 - 1e-3)
        with torch.no_grad():
            norm = torch.clamp((X * X).sum(1), max=1 - 1e-4)
            u, v = X[I], X[J]
            diff2 = ((u - v) ** 2).sum(1)
            arg = 1 + 2 * diff2 / ((1 - norm[I]) * (1 - norm[J]) + eps)
            dP = torch.acosh(torch.clamp(arg, min=1 + eps)).cpu().numpy()
        hyp = float(spearmanr(D[iu], dP).correlation)
        # Euclidean PCA baseline on the SAME sample and dimension
        Vc = self.V[idx] - self.V[idx].mean(0)
        U, S, Wt = np.linalg.svd(Vc, full_matrices=False)
        C = Vc @ Wt[:dim].T
        Demb = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        euc = float(spearmanr(D[iu], Demb[iu]).correlation)
        return {"dim": dim, "hyperbolic_spearman": round(hyp, 4),
                "euclidean_spearman": round(euc, 4),
                "hyperbolic_wins": bool(hyp > euc),
                "improvement": round(hyp - euc, 4)}

    def _blend_descriptor(self, smiles_list):
        """Mean descriptor profile of a blend; each molecule uses its own descriptors if
        known, else the nearest known molecule's (structural NN)."""
        vecs = []
        for s in smiles_list:
            try:
                f = smiles_to_ecfp4(s)
            except Exception:
                continue
            sims = (self.FP @ f) / (np.linalg.norm(self.FP, axis=1) * np.linalg.norm(f) + 1e-9)
            vecs.append(self.V[int(np.argmax(sims))])
        return np.mean(np.vstack(vecs), axis=0) if vecs else None

    def _gen_jaccard_to_all(self, prof):
        """Generalized (weighted) Jaccard distance from a fractional profile to every known
        molecule's binary descriptor vector: 1 - sum(min)/sum(max). Works in the true
        perceptual descriptor space, so it does not depend on the (lossy) linear embedding."""
        B = (self.V > 0).astype(float)
        p = np.clip(prof, 0, 1)[None, :]
        mn = np.minimum(p, B).sum(1)
        mx = np.maximum(p, B).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(mx > 0, mn / mx, 0.0)
        return 1.0 - sim

    def locate(self, smiles_list, k=8):
        """Place a blend in flavor space. Nearest flavors and the novelty hypothesis are
        computed in the FULL perceptual (descriptor) space via generalized Jaccard; the
        low-dim PCA coordinate is reported only as a rough map position (the linear
        embedding is lossy, consistent with flavor space being non-Euclidean)."""
        prof = self._blend_descriptor(smiles_list)
        if prof is None:
            return None
        coord = (prof - self._mean) @ self.components.T
        d = self._gen_jaccard_to_all(prof)
        order = np.argsort(d)[:k]
        knn_mean = float(d[order].mean())
        # novelty percentile: compare this blend's isolation to that of RANDOM BLENDS of the
        # SAME size (apples-to-apples). A blend of m molecules has a spread-out profile, so
        # the baseline must also be m-molecule blends, not single molecules.
        m = max(1, len(smiles_list))
        rng = np.random.default_rng(0)
        Bfull = (self.V > 0).astype(float)
        n = len(self.names)
        base = []
        for _ in range(300):
            idx = rng.choice(n, size=min(m, n), replace=False)
            bp = Bfull[idx].mean(0)
            db = self._gen_jaccard_to_all(bp)
            base.append(np.sort(db)[:k].mean())
        pct = float((np.array(base) < knn_mean).mean())
        return {"map_coordinates_pca": [round(float(x), 3) for x in coord],
                "nearest_known_flavors": [self.names[i] for i in order[:5]],
                "local_isolation_knn_dist": round(knn_mean, 3),
                "novelty_percentile": round(pct, 3),
                "interpretation": ("sits in a sparse/underexplored region of flavor space (novel)"
                                   if pct > 0.75 else "sits among known flavors")}


if __name__ == "__main__":
    import json
    g = FlavorSpaceGeometry(dim=3)
    print("molecules:", len(g.names))
    print("hyperbolicity:", json.dumps(g.hyperbolicity()))
    print("embedding faithfulness (Spearman):", g.embedding_faithfulness())
    # locate a de-novo-ish vanilla/fruity blend
    demo = ["O=Cc1ccc(O)c(OC)c1", "CCCCCC(=O)OCC", "CC(C)=CCCC(C)(O)C=C"]
    print("locate demo:", json.dumps(g.locate(demo)))
