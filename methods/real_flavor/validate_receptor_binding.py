"""Held-out validation of the Boltz receptor-binding layer: does a structure-based co-fold
recover receptor SPECIFICITY for molecules whose receptor labels are not in this repo?

Design (a specificity "swap" test on two well-characterized human ORs):

  OR5AN1  is the muscone / macrocyclic-musk receptor (Shirasu et al. 2014, Neuron).
  OR51E2  is the short-chain fatty-acid (propionate/acetate) receptor (Saito/Fujita).

  Ligands: two macrocyclic musks (muscone, civetone) and two short-chain acids
  (propionic, acetic). For each receptor, its COGNATE class should score higher than the
  NON-cognate class:
      OR5AN1: musks (cognate)  >  acids (non-cognate)
      OR51E2: acids (cognate)  >  musks (non-cognate)

Why this is a fair test: these molecule->receptor assignments come from external pharmacology,
NOT from Keller/Mainland labels used elsewhere here, and Boltz has never seen this repo's data.
So a correct cognate>non-cognate ranking is a genuine out-of-training-set prediction of a known
"tasty/known" pairing, exactly the claim we want to support (or refute).

Honesty: odorant receptors are 7TM GPCRs and Boltz-2's affinity/binding head is trained mostly
on soluble complexes, so we do NOT assume it works. This script reports the per-pair scores, a
per-receptor cognate-vs-non-cognate margin, and a separation AUC. If it fails to separate, that
is the finding, and the structural-NN layer in receptors.py stays the receptor grounding.

Requires Boltz credits (api.boltz.bio) and an OAuth session. Results cache under
results/boltz/, so this is cheap to re-run once the co-folds exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import receptor_binding as rb

HERE = Path(__file__).resolve().parent

LIGANDS = {
    "muscone":        ("O=C1CCCCCCCCCCCCC(C)C1", "musk"),
    "civetone":       ("O=C1CCCCCCCC=CCCCCCCC1", "musk"),
    "propionic_acid": ("CCC(=O)O", "acid"),
    "acetic_acid":    ("CC(=O)O", "acid"),
}
# receptor -> the ligand CLASS it is cognate for
COGNATE = {"OR5AN1": "musk", "OR51E2": "acid"}

# binding-confidence-like fields, best-first; first present is used as the score.
SCORE_FIELDS = ["binding_confidence", "complex_iptm", "iptm",
                "affinity_probability_binary", "confidence_score", "ptm"]


def _score(metrics):
    for k in SCORE_FIELDS:
        if k in metrics and metrics[k] is not None:
            return k, float(metrics[k])
    return None, None


def _auc(pos, neg):
    """Rank AUC = P(random cognate score > random non-cognate score)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def run(estimate_only=False):
    # cost check first
    est = rb.estimate_cost("OR51E2", "CCC(=O)O")
    per = est.get("estimated_cost_usd")
    n = len(rb.RECEPTORS) * len(LIGANDS)
    print(f"co-folds: {n}  (~${per} each; ~${float(per)*n:.2f} total)" if per else f"co-folds: {n}")
    if estimate_only:
        return {"estimate_usd_each": per, "n_cofolds": n}

    rows, score_field = [], None
    for rec in rb.RECEPTORS:
        for lig, (smi, cls) in LIGANDS.items():
            r = rb.cofold(rec, smi)
            fld, sc = _score(r["metrics"])
            score_field = score_field or fld
            rows.append({"receptor": rec, "ligand": lig, "class": cls,
                         "cognate": cls == COGNATE[rec], "score": sc,
                         "score_field": fld, "cached": r.get("cached")})
            tag = "COGNATE" if cls == COGNATE[rec] else "non-cog"
            print(f"  {rec}  {lig:16s} [{tag}]  score={sc}")

    # per-receptor margin + overall separation
    per_receptor, all_pos, all_neg = {}, [], []
    for rec in rb.RECEPTORS:
        pos = [r["score"] for r in rows if r["receptor"] == rec and r["cognate"] and r["score"] is not None]
        neg = [r["score"] for r in rows if r["receptor"] == rec and not r["cognate"] and r["score"] is not None]
        all_pos += pos; all_neg += neg
        margin = (round(sum(pos)/len(pos) - sum(neg)/len(neg), 4)
                  if pos and neg else None)
        per_receptor[rec] = {"cognate_mean": round(sum(pos)/len(pos), 4) if pos else None,
                             "noncognate_mean": round(sum(neg)/len(neg), 4) if neg else None,
                             "margin": margin, "correct_direction": margin is not None and margin > 0}
    auc = _auc(all_pos, all_neg)
    correct = sum(1 for v in per_receptor.values() if v["correct_direction"])
    out = {"score_field": score_field, "rows": rows, "per_receptor": per_receptor,
           "separation_auc": auc,
           "receptors_correct_direction": f"{correct}/{len(per_receptor)}",
           "verdict": ("SEPARATES" if (auc or 0) > 0.5 and correct == len(per_receptor)
                       else "DOES NOT SEPARATE (keep NN layer)")}
    print(f"\nscore field: {score_field}")
    for rec, v in per_receptor.items():
        print(f"  {rec}: cognate {v['cognate_mean']} vs non-cognate {v['noncognate_mean']} "
              f"margin {v['margin']} {'OK' if v['correct_direction'] else 'X'}")
    print(f"separation AUC (cognate>non-cognate): {auc}")
    print(f"verdict: {out['verdict']}")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "receptor_binding.json").write_text(json.dumps(out, indent=2))
    print("wrote results/receptor_binding.json")
    return out


if __name__ == "__main__":
    import sys
    run(estimate_only="--estimate" in sys.argv)
