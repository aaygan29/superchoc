"""Structure-based receptor-interaction layer: co-fold a flavor molecule against a real
odorant/taste receptor and read out a binding signal (Boltz-2).

Motivation. Wintermute's chain is molecule -> receptor activation -> percept, and receptors.py
grounds the middle only through a sparse *structural nearest-neighbour* to Mainland 2015
confirmed OR-ligand pairs (a ~1.6x-chance signal). That is the weakest quantitative link in
the pipeline: it never looks at the receptor's structure, only at ligand-ligand similarity.

This module co-folds the ligand INTO the receptor with Boltz-2 (api.boltz.bio) and returns the
predicted complex confidence (ipTM, the interface predicted-TM) plus the affinity head, giving
a genuine structure-based interaction score for a molecule--receptor pair. It is zero-shot with
respect to this repo: Boltz has never seen the Keller/Mainland labels, so a molecule that is
not in any training set here can still be scored against a receptor.

Honesty. Odorant receptors are 7-transmembrane GPCRs and Boltz-2's affinity head was trained
largely on soluble protein-ligand complexes; membrane GPCRs with tiny volatile ligands are out
of its comfort zone. So this layer must be *validated*, not trusted: validate_receptor_binding.py
checks whether the Boltz score actually separates KNOWN cognate ligands from non-cognate ones
for the same receptor. If it does not discriminate, we report that honestly and keep the NN
layer; if it does, it upgrades the receptor grounding.

Costs money and GPU time per call, so every run is cached by input hash under results/boltz/.
Always estimate-cost before a batch.

CLI: boltz-api (see memory reference_boltz_cli_setup). Requires an OAuth session
(`boltz-api auth login`).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "results" / "boltz"
RUNROOT = HERE / "results" / "boltz" / "_runs"
MODEL = "boltz-2.1"

# Real human odorant/taste receptor sequences (UniProt, retrieved 2026-09-11).
RECEPTORS = {
    "OR5AN1": {  # muscone / macrocyclic-musk receptor (Shirasu et al. 2014, Neuron)
        "uniprot": "Q8NGI8",
        "seq": ("MTGGGNITEITYFILLGFSDFPRIIKVLFTIFLVIYITSLAWNLSLIVLIRMDSHLHTPM"
                "YFFLSNLSFIDVCYISSTVPKMLSNLLQEQQTITFVGCIIQYFIFSTMGLSESCLMTAMA"
                "YDRYAAICNPLLYSSIMSPTLCVWMVLGAYMTGLTASLFQIGALLQLHFCGSNVIRHFFC"
                "DMPQLLILSCTDTFFVQVMTAILTMFFGIASALVIMISYGYIGISIMKITSAKGRSKAFN"
                "TCASHLTAVSLFYTSGIFVYLSSSSGGSSSFDRFASVFYTVVIPMLNPLIYSLRNKEIKD"
                "ALKRLQKRKCC"),
    },
    "OR51E2": {  # short-chain fatty-acid (propionate/acetate) receptor (Fujita/Saito)
        "uniprot": "Q9H255",
        "seq": ("MSSCNFTHATFVLIGIPGLEKAHFWVGFPLLSMYVVAMFGNCIVVFIVRTERSLHAPMYL"
                "FLCMLAAIDLALSTSTMPKILALFWFDSREISFEACLTQMFFIHALSAIESTILLAMAFD"
                "RYVAICHPLRHAAVLNNTVTAQIGIVAVVRGSLFFFPLPLLIKRLAFCHSNVLSHSYCVH"
                "QDVMKLAYADTLPNVVYGLTAILLVMGVDVMFISLSYFLIIRTVLQLPSKSERAKAFGTC"
                "VSHIGVVLAFYVPLIGLSVVHRFGNSLHPIVRVVMGDIYLLLPPVINPIIYGAKTKQIRT"
                "RVLAMFKISCDKDLQAVGGK"),
    },
}


def _key(receptor, smiles):
    h = hashlib.sha256(f"{MODEL}|{receptor}|{smiles}".encode()).hexdigest()[:16]
    return f"{receptor}__{h}"


def _write_yaml(seq, smiles, path):
    # newline-free single-line SMILES; quote to be safe. Binding metrics are requested via a
    # top-level `binding` block (api.boltz.bio schema), not a per-entity/`properties` field.
    y = (
        "entities:\n"
        "  - type: protein\n"
        f"    value: {seq}\n"
        '    chain_ids: ["A"]\n'
        "  - type: ligand_smiles\n"
        f"    value: \"{smiles}\"\n"
        '    chain_ids: ["B"]\n'
        "binding:\n"
        '  type: "ligand_protein_binding"\n'
        '  binder_chain_id: "B"\n'
    )
    Path(path).write_text(y)


def _boltz(args, timeout=1200):
    env = dict(os.environ)
    env["PATH"] = f"{os.path.expanduser('~')}/.local/bin:" + env.get("PATH", "")
    return subprocess.run(["boltz-api", *args], capture_output=True, text=True,
                          env=env, timeout=timeout)


def estimate_cost(receptor, smiles):
    r = RECEPTORS[receptor]
    yaml = RUNROOT / f"{_key(receptor, smiles)}.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(r["seq"], smiles, yaml)
    p = _boltz(["predictions:structure-and-binding", "estimate-cost",
                "--model", MODEL, "--format", "json", "--input", f"@yaml://{yaml}"])
    try:
        return json.loads(p.stdout)
    except Exception:
        return {"raw": p.stdout, "err": p.stderr}


def _parse_metrics(run_dir: Path):
    """Pull confidence + affinity from a completed run's output tree.

    Boltz-2 writes metrics.json with the per-sample metrics nested under best_sample.metrics
    (ptm, iptm, ligand_iptm, structure_confidence, complex_plddt, ...). Affinity, when present,
    appears as affinity_pred_value / affinity_probability_binary. We flatten whichever exist.
    """
    wanted = ("structure_confidence", "ptm", "iptm", "ligand_iptm", "protein_iptm",
              "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
              "affinity_pred_value", "affinity_probability_binary")
    out = {}
    for f in run_dir.rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        # search top-level and one nested level (best_sample.metrics)
        candidates = [d]
        if isinstance(d, dict):
            bs = d.get("best_sample")
            if isinstance(bs, dict) and isinstance(bs.get("metrics"), dict):
                candidates.append(bs["metrics"])
            if isinstance(d.get("metrics"), dict):
                candidates.append(d["metrics"])
        for c in candidates:
            if not isinstance(c, dict):
                continue
            for k in wanted:
                if k in c and c[k] is not None:
                    out[k] = c[k]
    return out


def cofold(receptor, smiles, use_cache=True, timeout=1800):
    """Run (or load cached) Boltz co-fold. Returns {receptor, smiles, metrics, cached}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = _key(receptor, smiles)
    cache_file = CACHE / f"{key}.json"
    if use_cache and cache_file.exists():
        d = json.loads(cache_file.read_text())
        d["cached"] = True
        return d

    r = RECEPTORS[receptor]
    yaml = RUNROOT / f"{key}.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(r["seq"], smiles, yaml)
    run_dir = RUNROOT / key
    p = _boltz(["predictions:structure-and-binding", "run",
                "--model", MODEL, "--name", key,
                "--idempotency-key", f"superchoc-{key}",
                "--format", "json",
                "--input", f"@yaml://{yaml}",
                "--root-dir", str(run_dir)], timeout=timeout)
    metrics = _parse_metrics(run_dir)
    result = {"receptor": receptor, "uniprot": r["uniprot"], "smiles": smiles,
              "metrics": metrics, "cached": False,
              "stdout_tail": p.stdout[-400:], "stderr_tail": p.stderr[-400:]}
    if metrics:  # only cache real successes
        cache_file.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import sys
    rec = sys.argv[1] if len(sys.argv) > 1 else "OR51E2"
    smi = sys.argv[2] if len(sys.argv) > 2 else "CCC(=O)O"
    print("cost estimate:", json.dumps(estimate_cost(rec, smi), indent=2)[:600])
