"""Live ingestion from public online sources, cached to data/raw.

"Online" here means pull-on-demand from public web APIs, so the corpus in
docs/GROUND_TRUTH.md can be refreshed continuously rather than shipped statically. Only
the Python standard library is used (urllib), so this runs anywhere.

Sources wired here (all public, no key required):
  - ChEMBL REST API: GPCR / olfactory-receptor targets and their bioactivities.
  - NCBI E-utilities (PubMed): literature for the cross-disciplinary library.
  - bioRxiv API: recent preprints by date range.

Each fetch returns parsed JSON and, if cache_dir is given, writes it there with a
descriptive filename. Nothing is fabricated: on a network or HTTP error the function
raises, it does not invent records.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

_UA = {"User-Agent": "superchoc-online-ingest/0.1 (research; contact via repo)"}


def _get_json(url: str, timeout: float = 30.0, retries: int = 3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # network/HTTP/JSON
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}\n{last}")


def _maybe_cache(obj, cache_dir, filename):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, filename)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        return path
    return None


# --- ChEMBL ---
def chembl_targets(query="olfactory receptor", limit=50, cache_dir=None):
    q = urllib.parse.quote(query)
    url = (f"https://www.ebi.ac.uk/chembl/api/data/target/search?q={q}"
           f"&format=json&limit={limit}")
    data = _get_json(url)
    targets = data.get("targets", [])
    _maybe_cache(targets, cache_dir, f"chembl_targets_{q}.json")
    return targets


def chembl_bioactivities(target_chembl_id, limit=200, cache_dir=None):
    url = (f"https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id="
           f"{target_chembl_id}&format=json&limit={limit}")
    data = _get_json(url)
    acts = data.get("activities", [])
    _maybe_cache(acts, cache_dir, f"chembl_activities_{target_chembl_id}.json")
    return acts


# --- PubMed (NCBI E-utilities) ---
def pubmed_search(term, retmax=20, cache_dir=None):
    q = urllib.parse.quote(term)
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed"
           f"&term={q}&retmode=json&retmax={retmax}")
    data = _get_json(url)
    ids = data.get("esearchresult", {}).get("idlist", [])
    _maybe_cache(ids, cache_dir, f"pubmed_ids_{q[:40]}.json")
    return ids


# --- bioRxiv ---
def biorxiv_recent(start, end, server="biorxiv", cache_dir=None):
    url = f"https://api.biorxiv.org/details/{server}/{start}/{end}/0"
    data = _get_json(url)
    coll = data.get("collection", [])
    _maybe_cache(coll, cache_dir, f"biorxiv_{server}_{start}_{end}.json")
    return coll


if __name__ == "__main__":
    # Smoke test against the live APIs.
    t = chembl_targets(limit=5)
    print("ChEMBL targets:", len(t), "e.g.", t[0].get("pref_name") if t else None)
    ids = pubmed_search("cocoa aroma key odorants", retmax=5)
    print("PubMed ids:", ids)
