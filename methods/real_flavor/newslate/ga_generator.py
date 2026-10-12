"""NEWSLATE (g, cont.): a genetic-algorithm generator as an ALTERNATIVE METHOD CLASS to BRICS.

BRICS (lego_assembly) is one-shot fragment enumeration. This is an evolutionary optimiser over
molecular graphs: a SELFIES-string genetic algorithm (Krenn 2020; cf. STONED/JANUS, Nigam 2021).
SELFIES is used because every mutation/crossover decodes to a valid molecule, so the GA is robust.

Head-to-head under the SAME criteria and SAME scorers as generator_baselines.py:
  - predictor A  : RandomForest on ECFP4 (the model the GA optimises; selection-biased for it)
  - predictor B  : descriptor GBM, INDEPENDENT of A -> the fair, non-circular comparison
  - novelty      : max ECFP4 Tanimoto to any Keller molecule < 0.5
  - safety       : safety.screen verdict OK
  - SA score     : Ertl & Schuffenhauer 2009 (synthetic accessibility)
  - joint pass   : predA >= Keller 75th pct AND novel AND SA <= 4.5

We report GA top-40 (by predA) vs the BRICS guided top-40 and vs unguided draws. The honest test
is predictor B (independent): if the GA only wins on A, that is selection on A, not better molecules.

Run:  python3 newslate/ga_generator.py        (writes results/newslate/ga_generator.json)
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import selfies as sf
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import Descriptors, inchi, rdFingerprintGenerator
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from real_data import build_keller_pleasantness
from safety import screen

OUT = os.path.abspath(os.path.join(HERE, "..", "results", "newslate"))
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(s):
    m = Chem.MolFromSmiles(s)
    return gen.GetFingerprint(m) if m is not None else None


def desc(s):
    d = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s))
    return np.array(list(d.values()), float)


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    if m.GetNumHeavyAtoms() < 3 or m.GetNumHeavyAtoms() > 25:
        return None          # flavor molecules are small; keep the search in that regime
    if Descriptors.MolWt(m) > 350:
        return None
    return Chem.MolToSmiles(m)


class GA:
    """SELFIES genetic algorithm. Fitness = predicted pleasantness with hard novelty+safety gates
    expressed as penalties, so selection is pushed toward novel, safe, pleasant molecules."""

    def __init__(self, seeds, score_fn, novel_fn, safe_fn, alphabet, seed=0,
                 pop=120, generations=40, p_mut=0.9, elite=0.15):
        self.rng = np.random.default_rng(seed)
        self.score_fn, self.novel_fn, self.safe_fn = score_fn, novel_fn, safe_fn
        self.alphabet = list(alphabet)
        self.pop, self.generations, self.p_mut, self.elite = pop, generations, p_mut, elite
        self.cache = {}
        self.population = [self._tokens(s) for s in seeds]
        while len(self.population) < pop:                     # pad by mutating seeds
            self.population.append(self._mutate(self.population[self.rng.integers(len(seeds))]))
        self.seen = {}                                        # smiles -> fitness (feasible only)

    def _tokens(self, smi):
        return list(sf.split_selfies(sf.encoder(smi)))

    def _decode(self, toks):
        try:
            return canon(sf.decoder("".join(toks)))
        except Exception:
            return None

    def _mutate(self, toks):
        toks = list(toks)
        op = self.rng.integers(3)
        if not toks:
            return [self.rng.choice(self.alphabet)]
        i = int(self.rng.integers(len(toks)))
        if op == 0:                                           # substitute
            toks[i] = self.rng.choice(self.alphabet)
        elif op == 1:                                         # insert
            toks.insert(i, self.rng.choice(self.alphabet))
        elif len(toks) > 1:                                   # delete
            toks.pop(i)
        return toks

    def _crossover(self, a, b):
        if not a or not b:
            return list(a or b)
        ca = int(self.rng.integers(1, len(a) + 1)) if len(a) > 1 else 1
        cb = int(self.rng.integers(1, len(b) + 1)) if len(b) > 1 else 1
        return a[:ca] + b[cb:]

    def _fitness(self, toks):
        smi = self._decode(toks)
        if smi is None:
            return -1e6, None
        if smi in self.cache:
            f = self.cache[smi]
        else:
            pleas = self.score_fn(smi)                        # 0..100
            pen = 0.0
            feasible = True
            if not self.novel_fn(smi):
                pen += 60.0; feasible = False
            if not self.safe_fn(smi):
                pen += 60.0; feasible = False
            f = pleas - pen
            self.cache[smi] = f
            if feasible:
                self.seen[smi] = pleas
        return f, smi

    def run(self):
        n_elite = max(2, int(self.elite * self.pop))
        for _ in tqdm(range(self.generations), desc="GA generations"):
            scored = sorted(((self._fitness(t)[0], t) for t in self.population),
                            key=lambda x: x[0], reverse=True)
            elites = [t for _, t in scored[:n_elite]]
            new = list(elites)
            while len(new) < self.pop:
                p1 = elites[self.rng.integers(len(elites))]
                p2 = scored[self.rng.integers(len(scored))][1]
                child = self._crossover(p1, p2)
                if self.rng.random() < self.p_mut:
                    child = self._mutate(child)
                new.append(child)
            self.population = new
        return self.seen                                      # feasible smiles -> pleasantness


def main():
    rng = np.random.default_rng(0)
    df = build_keller_pleasantness().reset_index(drop=True)
    kfps = [fp(s) for s in df.smiles]
    y = df.pleasantness.values
    Xe = np.array([list(f) for f in kfps], dtype=np.int8)
    modelA = RandomForestRegressor(300, random_state=0, n_jobs=-1).fit(Xe, y)   # same as baselines
    D = np.clip(np.nan_to_num(np.vstack([desc(s) for s in df.smiles]), posinf=0, neginf=0), -1e6, 1e6)
    modelB = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, random_state=0).fit(D, y)
    pb = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(D):
        pb[te] = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                           subsample=0.8, random_state=0).fit(D[tr], y[tr]).predict(D[te])
    q75 = float(np.quantile(y, 0.75))
    known_ik = {inchi.MolToInchiKey(Chem.MolFromSmiles(s)) for s in df.smiles}
    safe_known = [s for s in df.smiles if screen(s)["verdict"] == "OK"]

    def predA(smi):
        f = fp(smi)
        return float(modelA.predict(np.array([list(f)], dtype=np.int8))[0]) if f is not None else 0.0

    def is_novel(smi):
        f = fp(smi)
        if f is None:
            return False
        if inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) in known_ik:
            return False
        return max(DataStructs.BulkTanimotoSimilarity(f, kfps)) < 0.5

    def is_safe(smi):
        try:
            return screen(smi)["verdict"] == "OK"
        except Exception:
            return False

    # alphabet: tokens seen in the safe Keller seeds + a robust default (keeps atoms flavor-like)
    alpha = set()
    for s in safe_known:
        try:
            alpha |= set(sf.split_selfies(sf.encoder(s)))
        except Exception:
            pass
    alpha |= set(sf.get_semantic_robust_alphabet())
    alpha = sorted(a for a in alpha if all(x not in a for x in ("P", "S", "Cl", "Br", "I", "F", "+", "-")))

    seeds = [safe_known[i] for i in rng.choice(len(safe_known), min(120, len(safe_known)), replace=False)]
    ga = GA(seeds, predA, is_novel, is_safe, alpha, seed=0, pop=120, generations=40)
    feasible = ga.run()                                      # novel + safe smiles -> predA

    # score helpers (identical to generator_baselines.metrics)
    def metrics(smis):
        smis = list(smis)
        Xs = np.array([list(fp(s)) for s in smis], dtype=np.int8)
        pa = modelA.predict(Xs)
        Ds = np.clip(np.nan_to_num(np.vstack([desc(s) for s in smis]), posinf=0, neginf=0), -1e6, 1e6)
        pb_ = modelB.predict(Ds)
        mt = np.array([max(DataStructs.BulkTanimotoSimilarity(fp(s), kfps)) for s in smis])
        sa = np.array([sascorer.calculateScore(Chem.MolFromSmiles(s)) for s in smis])
        ok = (pa >= q75) & (mt < 0.5) & (sa <= 4.5)
        return {"n": len(smis), "pred_pleasantness_A": float(pa.mean()),
                "pred_pleasantness_B_indep": float(pb_.mean()),
                "mean_max_tanimoto_to_keller": float(mt.mean()), "mean_SA": float(sa.mean()),
                "median_SA": float(np.median(sa)), "frac_SA_gt_6": float((sa > 6).mean()),
                "frac_SA_le_4.5": float((sa <= 4.5).mean()), "frac_joint_pass": float(ok.mean()),
                "frac_pass_pleasant_A": float((pa >= q75).mean()), "frac_pass_novel": float((mt < 0.5).mean())}

    ga_smis = list(feasible.keys())
    res = {
        "method": "SELFIES genetic algorithm (evolutionary) vs BRICS enumeration",
        "ga_config": {"pop": 120, "generations": 40, "p_mut": 0.9, "elite": 0.15, "seed": 0,
                      "alphabet_size": len(alpha), "n_seeds": len(seeds)},
        "criteria": {"q75_keller_pleasantness": q75, "novel": "max ECFP4 Tanimoto to Keller < 0.5",
                     "safe": "safety.screen OK", "SA_pass": "<= 4.5"},
        "modelB_random_cv_spearman": float(spearmanr(y, pb).correlation),
        "ga_n_feasible_unique": len(ga_smis),
    }
    if ga_smis:
        res["ga_all_feasible"] = metrics(ga_smis)
        top = sorted(ga_smis, key=lambda s: feasible[s], reverse=True)[:40]
        res["ga_guided_top40_by_predA"] = metrics(top)
        res["ga_top40_example_smiles"] = top[:10]

    # BRICS head-to-head: re-score the guided top-40 from lego_denovo.json with the SAME scorers
    lego = json.load(open(os.path.join(os.path.dirname(HERE), "results", "lego_denovo.json")))
    cands = [c["smiles"] for c in lego["candidates"]]
    Xc = np.array([list(fp(s)) for s in cands], dtype=np.int8)
    pa_c = modelA.predict(Xc)
    order = np.argsort(-pa_c)
    brics_top = [cands[i] for i in order[:40]]
    res["brics_guided_top40_by_predA"] = metrics(brics_top)
    res["brics_all_candidates"] = metrics(cands)

    # the fair, non-circular read: compare on the INDEPENDENT model B
    if ga_smis:
        res["headline_independent_modelB"] = {
            "ga_top40": res["ga_guided_top40_by_predA"]["pred_pleasantness_B_indep"],
            "brics_top40": res["brics_guided_top40_by_predA"]["pred_pleasantness_B_indep"],
            "ga_minus_brics": res["ga_guided_top40_by_predA"]["pred_pleasantness_B_indep"]
                              - res["brics_guided_top40_by_predA"]["pred_pleasantness_B_indep"],
            "note": "positive = GA finds more pleasant molecules than BRICS under a model it did NOT optimise",
        }

    json.dump(res, open(os.path.join(OUT, "ga_generator.json"), "w"), indent=2, default=float)
    printable = {k: v for k, v in res.items() if k != "ga_top40_example_smiles"}
    print(json.dumps(printable, indent=1, default=float))


if __name__ == "__main__":
    main()
