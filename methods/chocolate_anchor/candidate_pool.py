"""Molecule pool for a prospective tasting study of new binary odor mixtures near chocolate.

Output: data/candidates/pool.csv (one row per molecule) + data/candidates/evidence.json
(the raw regulatory evidence the CSV was derived from, with the URLs that were checked).

Eligibility is conservative: a molecule is eligible only if
  1. a FEMA GRAS number is verified on the FEMA flavor library page for that exact CAS, or the
     CAS is on the EU Union List (Reg. 872/2012 annex, adopting Reg. 1334/2008 Annex I Part A);
  2. it is not one of the substances FDA removed in 83 FR 50490 (2018-10-09);
  3. it is not in the Ma 2021 data set (those mixtures are our test data);
  4. safety.screen is not TOXIC/INVALID;
  5. the Union List row carries no "Restrictions of Use".
Anything unverifiable is ineligible ("regulatory status unverified"); nothing is assumed. A FEMA page found only by
name (page lists a parent CAS) and matched to PubChem's FEMA annotation is not a verification either.

GRAS/Union-List status covers use as a flavouring at customary levels. It is not a statement
that a panel stimulus is safe at any concentration.

Run:  .venv/bin/python -m methods.chocolate_anchor.candidate_pool [--refresh]
Without --refresh the committed evidence.json is reused (offline, deterministic).
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem

from methods.real_flavor.safety import screen

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "candidates"
POOL_CSV = OUT_DIR / "pool.csv"
EVIDENCE_JSON = OUT_DIR / "evidence.json"

FEMA_SEARCH = "https://www.femaflavor.org/flavor-library/search?fulltext="
FEMA_PAGE = "https://www.femaflavor.org/flavor-library/"
EU_872_URL = "http://publications.europa.eu/resource/celex/32012R0872"  # Cellar, xhtml
EU_872_ELI = "https://eur-lex.europa.eu/eli/reg_impl/2012/872/oj"
FDA_FR_URL = "https://www.federalregister.gov/documents/2018/10/09/2018-21807/food-additive-regulations-synthetic-flavoring-agents-and-adjuvants"
FDA_FR_XML = "https://www.federalregister.gov/documents/full_text/xml/2018/10/09/2018-21807.xml"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound"

# Additional confectionery odorants (name, CAS). Each still has to pass the regulatory check.
# The last four are FDA-2018-delisted substances kept on purpose to exercise the exclusion.
ADDED: list[tuple[str, str]] = [
    ("ethyl vanillin", "121-32-4"), ("ethyl maltol", "4940-11-8"), ("maltol", "118-71-8"),
    ("methyl cyclopentenolone (cyclotene)", "80-71-7"), ("delta-decalactone", "705-86-2"),
    ("delta-octalactone", "698-76-0"), ("gamma-undecalactone", "104-67-6"),
    ("delta-dodecalactone", "713-95-1"), ("diacetyl", "431-03-8"), ("acetoin", "513-86-0"),
    ("ethyl hexanoate", "123-66-0"), ("ethyl octanoate", "106-32-1"), ("ethyl butanoate", "105-54-4"),
    ("hexyl acetate", "142-92-7"), ("benzyl acetate", "140-11-4"), ("methyl anthranilate", "134-20-3"),
    ("alpha-ionone", "127-41-3"), ("beta-ionone", "14901-07-6"), ("beta-damascenone", "23726-93-4"),
    ("raspberry ketone", "5471-51-2"), ("citral", "5392-40-5"), ("d-limonene", "5989-27-5"),
    ("2-acetylpyrazine", "22047-25-2"), ("2-acetylthiazole", "24295-03-2"),
    ("2,5-dimethylpyrazine", "123-32-0"), ("furfural", "98-01-1"), ("5-methylfurfural", "620-02-0"),
    ("methional", "3268-49-3"), ("2-furfurylthiol", "98-02-2"), ("geraniol", "106-24-1"),
    ("rose oxide", "16409-43-1"), ("cis-3-hexenol", "928-96-1"), ("nonanal", "124-19-6"),
    ("2-heptanone", "110-43-0"), ("eugenol", "97-53-0"), ("cinnamaldehyde", "104-55-2"),
    ("menthol", "89-78-1"), ("carvone", "99-49-0"), ("hexanoic acid", "142-62-1"),
    ("myrcene (FDA-delisted)", "123-35-3"),
]
DELISTED_DEMOS = [("pyridine (FDA-delisted)", "110-86-1"), ("pulegone (FDA-delisted)", "89-82-7")]


@dataclass(frozen=True)
class Molecule:
    name: str
    cas: str
    smiles: str | None
    pool_source: str


def _get(url: str, *, accept: str | None = None, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; candidate-pool)"})
    if accept:
        req.add_header("Accept", accept)
        req.add_header("Accept-Language", "eng")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _text(html: bytes) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", html.decode("utf8", "replace")))


def flat_smiles(smiles: str) -> str:
    """Stereo-free canonical SMILES, used to match molecules across data sets."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------- molecule universe

def load_universe() -> list[Molecule]:
    cocoa = pd.read_csv(ROOT / "data/sensomics/cocoa_key_odorants.csv")
    gras = pd.read_csv(ROOT / "data/sensomics/odor_thresholds_food_gras.csv")
    seen: dict[str, Molecule] = {}
    for df, src in ((cocoa, "cocoa_key_odorants.csv"), (gras, "odor_thresholds_food_gras.csv")):
        for _, r in df.drop_duplicates("cas").iterrows():
            if r.cas in seen:
                seen[r.cas] = Molecule(seen[r.cas].name, r.cas, seen[r.cas].smiles, seen[r.cas].pool_source + "+" + src)
            else:
                seen[r.cas] = Molecule(r["name"], r.cas, r.smiles, src)
    for name, cas in ADDED + DELISTED_DEMOS:
        if cas not in seen:
            seen[cas] = Molecule(name, cas, None, "added_confectionery")
    return list(seen.values())


# ---------------------------------------------------------------- evidence gathering

def fetch_pubchem(cas: str) -> dict:
    q = urllib.parse.quote(cas)
    cids = json.loads(_get(f"{PUBCHEM}/compound/name/{q}/cids/JSON"))["IdentifierList"]["CID"]
    for cid in cids:
        syn = json.loads(_get(f"{PUBCHEM}/compound/cid/{cid}/synonyms/JSON"))["InformationList"]["Information"][0]["Synonym"]
        if cas in syn:
            break
    else:
        raise ValueError(f"no PubChem CID lists CAS {cas} among its synonyms (candidates {cids})")
    props = json.loads(_get(f"{PUBCHEM}/compound/cid/{cid}/property/SMILES,Title/JSON"))["PropertyTable"]["Properties"][0]
    fema_nums: list[str] = []
    try:
        view = json.loads(_get(f"{PUBCHEM_VIEW}/{cid}/JSON?heading=FEMA+Number"))
        def walk(sec):
            if sec.get("TOCHeading") == "FEMA Number":
                for info in sec.get("Information", []):
                    for s in info["Value"]["StringWithMarkup"]:
                        fema_nums.append(s["String"])
            for sub in sec.get("Section", []):
                walk(sub)
        for sec in view["Record"]["Section"]:
            walk(sec)
    except Exception:  # heading absent for this CID -> no PubChem FEMA annotation
        pass
    return {"cid": cid, "smiles": props["SMILES"], "title": props["Title"],
            "pubchem_fema": sorted(set(fema_nums)),
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"}


def _fema_page(slug: str) -> dict | None:
    page = _text(_get(FEMA_PAGE + slug))
    m_fema = re.search(r"FEMA Number (\d+)", page)
    m_cas = re.search(r"\bCAS ((?:[0-9]+-[0-9]+-[0-9](?:; )?)+)", page)
    if not (m_fema and m_cas):
        return None
    m_cfr = re.search(r"CFR (21CFR[0-9.;\s]+?)(?: JECFA| SLR| Become|$)", page)
    return {"fema_number": int(m_fema.group(1)), "url": FEMA_PAGE + slug,
            "cas_listed": [c.strip() for c in m_cas.group(1).split(";") if c.strip()],
            "cfr": m_cfr.group(1).strip() if m_cfr else ""}


def fetch_fema(cas: str, name: str) -> dict:
    """Search the FEMA flavor library by CAS, then by name; accept only a page whose own CAS list
    contains `cas` (a name match alone is never accepted)."""
    searched, slugs_seen, name_candidates = [], [], []
    queries = [cas, re.sub(r"\s*\(.*\)", "", name), re.sub(r"^(alpha|beta|gamma|delta)-", "", name)]
    for q in dict.fromkeys(queries):
        url = FEMA_SEARCH + urllib.parse.quote(q)
        searched.append(url)
        html = _get(url).decode("utf8", "replace")
        for slug in sorted(set(re.findall(r'href="/flavor-library/([a-z0-9%\-_.]+)"', html)) - {"search"}):
            slugs_seen.append(slug)
            info = _fema_page(slug)
            if info and cas in info["cas_listed"]:
                return {"found": True, "matched_by": "CAS" if q == cas else f"name search '{q}', CAS confirmed on page", **info}
            if info and q != cas:
                name_candidates.append(info)
    return {"found": False, "searched_urls": searched, "slugs_seen": sorted(set(slugs_seen)),
            "name_only_candidates": name_candidates}


def parse_eu_union_list(xhtml: bytes) -> dict[str, list[dict]]:
    from lxml import etree

    root = etree.fromstring(xhtml, etree.HTMLParser())
    by_cas: dict[str, list[dict]] = {}
    for tr in root.xpath("//tr"):
        c = [" ".join("".join(td.itertext()).split()) for td in tr.xpath("./td")]
        if len(c) >= 9 and re.fullmatch(r"\d{2}\.\d{3}", c[0]) and re.fullmatch(r"\d+-\d+-\d", c[2]):
            by_cas.setdefault(c[2], []).append(
                {"fl_no": c[0], "name": c[1], "restrictions_of_use": c[6], "footnote": c[7], "reference": c[8]})
    return by_cas


FDA_DELISTED_NAMES = {
    "119-61-9": "benzophenone", "140-88-5": "ethyl acrylate", "93-15-2": "eugenyl methyl ether",
    "123-35-3": "myrcene", "89-82-7": "pulegone", "110-86-1": "pyridine", "100-42-5": "styrene",
}


def parse_fda_delisted(xml: bytes) -> dict[str, str]:
    """CAS -> name for the seven substances the notice enumerates as 'CAS No. ...'.

    Names are keyed by CAS (italic markup splits the pulegone line); the CAS set found in the
    notice must equal the expected seven, so a changed document fails loudly.
    """
    found = set(re.findall(r"CAS No\. ([0-9]+-[0-9]+-[0-9])", _text(xml)))
    assert found == set(FDA_DELISTED_NAMES), found
    return dict(FDA_DELISTED_NAMES)


def gather_evidence(universe: list[Molecule]) -> dict:
    eu_all = parse_eu_union_list(_get(EU_872_URL, accept="application/xhtml+xml"))
    fda = parse_fda_delisted(_get(FDA_FR_XML))
    mols = {}
    for m in universe:
        entry: dict = {}
        try:
            entry["pubchem"] = fetch_pubchem(m.cas)
        except Exception as e:
            entry["pubchem"] = {"error": str(e)}
        entry["fema"] = fetch_fema(m.cas, m.name)
        entry["eu"] = eu_all.get(m.cas, [])
        mols[m.cas] = entry
        time.sleep(0.3)
        print("evidence", m.cas, m.name, file=sys.stderr)
    return {
        "checked_on": time.strftime("%Y-%m-%d"),
        "sources": {
            "fema_search": FEMA_SEARCH + "<CAS>", "fema_page": FEMA_PAGE + "<slug>",
            "eu_union_list": f"{EU_872_ELI} (xhtml via {EU_872_URL}); Reg. 872/2012 as adopted, NOT the later consolidated text",
            "fda_delisting": FDA_FR_URL, "pubchem": "https://pubchem.ncbi.nlm.nih.gov/ (FEMA Number annotation, cross-check only)",
        },
        "fda_delisted_2018": fda,
        "molecules": mols,
    }


def load_evidence(refresh: bool, universe: list[Molecule]) -> dict:
    if refresh or not EVIDENCE_JSON.exists():
        ev = gather_evidence(universe)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        EVIDENCE_JSON.write_text(json.dumps(ev, indent=1, sort_keys=True))
        return ev
    return json.loads(EVIDENCE_JSON.read_text())


# ---------------------------------------------------------------- test-set / cocoa lookups

def ma_and_keller_keys() -> tuple[set[int], set[str], set[int]]:
    ma = pd.read_csv(ROOT / "data/hedonics/raw/ma_2021/molecules.csv")
    keller = pd.read_csv(ROOT / "data/hedonics/raw/keller_2016/keller_2016.csv", usecols=["CID"])
    return set(ma.CID), {flat_smiles(s) for s in ma.IsomericSMILES}, set(keller.CID)


def cocoa_key_table() -> dict[str, tuple[float, list[str]]]:
    d = pd.read_csv(ROOT / "data/sensomics/cocoa_key_odorants.csv")
    d["oav"] = pd.to_numeric(d["oav"], errors="coerce")
    d = d[d.oav >= 1]
    return {cas: (float(g.oav.max()), sorted(g.source_key.unique())) for cas, g in d.groupby("cas")}


# ---------------------------------------------------------------- assembly

COLUMNS = ["name", "cid", "smiles", "role", "oav_max", "oav_sources", "pool_source", "cas", "fema_number",
           "regulatory_basis", "eu_flavouring_id", "regulatory_evidence", "in_ma2021", "in_keller",
           "hazard_screen", "eligible", "exclusion_reason"]


def _fema_pubchem_crosswalk(e: dict) -> dict | None:
    """A name-found FEMA page whose number equals PubChem's FEMA annotation for the exact CID."""
    if e["fema"]["found"]:
        return None
    annotated = {str(x) for x in e["pubchem"].get("pubchem_fema", [])}
    hits = [c for c in e["fema"]["name_only_candidates"] if str(c["fema_number"]) in annotated]
    return hits[0] if len(hits) == 1 else None


def build_rows(ev: dict, universe: list[Molecule]) -> list[dict]:
    ma_cids, ma_flat, keller_cids = ma_and_keller_keys()
    key = cocoa_key_table()
    fda = ev["fda_delisted_2018"]
    rows = []
    for m in universe:
        e = ev["molecules"][m.cas]
        pc = e["pubchem"]
        reasons: list[str] = []
        smiles_src = m.smiles if m.smiles else pc.get("smiles")
        assert smiles_src, f"no SMILES for {m.name}"
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_src))
        cid = pc.get("cid", "")
        in_ma = (cid in ma_cids) or (flat_smiles(canon) in ma_flat)
        in_keller = cid in keller_cids if cid != "" else False

        basis, evid = [], []
        fema_no = ""
        alt = _fema_pubchem_crosswalk(e)
        if e["fema"]["found"]:
            fema_no = e["fema"]["fema_number"]
            basis.append("FEMA GRAS")
            evid.append(f"FEMA library page (CAS match) {e['fema']['url']} FEMA {fema_no}"
                        + (f"; {e['fema']['cfr']}" if e["fema"]["cfr"] else ""))
        elif alt is not None:
            # Library entry found by name lists a parent CAS (e.g. isomer-unspecified) and PubChem's FEMA annotation
            # on this exact CID carries the same number. Recorded as evidence, but it is NOT a verification: the
            # rule needs a FEMA page listing this CAS or an EU row for it.
            evid.append(f"parent-CAS/annotation only (not accepted): FEMA library page found by name (page CAS "
                        f"{', '.join(alt['cas_listed'])} differs from {m.cas}) {alt['url']} FEMA {alt['fema_number']}, "
                        f"matching PubChem annotation on CID {cid}")
        if pc.get("pubchem_fema"):
            evid.append(f"PubChem annotation {', '.join(pc['pubchem_fema'])} ({pc['pubchem_url']})")
            if fema_no != "" and not any(str(fema_no) in x for x in pc["pubchem_fema"]):
                evid.append("NOTE: PubChem FEMA annotation disagrees with FEMA library")
        eu = e["eu"]
        eu_ids = ";".join(r["fl_no"] for r in eu)
        eu_restrict = "; ".join(r["restrictions_of_use"] for r in eu if r["restrictions_of_use"])
        if eu:
            basis.append("EU Union List")
            evid.append(f"EU Reg. 872/2012 annex row(s) FL {eu_ids} for CAS {m.cas} ({EU_872_ELI})")

        verified = bool(basis)
        if m.cas in fda:
            reasons.append(f"FDA removed 2018 ({fda[m.cas]}; 83 FR 50490, {FDA_FR_URL})")
        if in_ma:
            reasons.append("in Ma 2021 data set (test data)")
        if not verified:
            reasons.append("regulatory evidence only via parent CAS / PubChem annotation" if alt is not None
                           else "regulatory status unverified")
        hz = screen(canon)
        if hz["verdict"] in ("TOXIC", "INVALID"):
            reasons.append(f"hazard screen {hz['verdict']}")
        if eu_restrict:
            reasons.append(f"EU Union List restriction of use: {eu_restrict}")

        oav = key.get(m.cas)
        rows.append({
            "name": m.name, "cid": cid, "smiles": canon,
            "role": "cocoa_key" if oav else "partner",
            "oav_max": oav[0] if oav else "", "oav_sources": ";".join(oav[1]) if oav else "",
            "pool_source": m.pool_source, "cas": m.cas, "fema_number": fema_no,
            "regulatory_basis": "; ".join(basis), "eu_flavouring_id": eu_ids,
            "regulatory_evidence": " | ".join(evid),
            "in_ma2021": in_ma, "in_keller": in_keller,
            "hazard_screen": json.dumps(hz, sort_keys=True),
            "eligible": not reasons, "exclusion_reason": "; ".join(reasons),
        })
    return rows


def main(refresh: bool = False) -> None:
    universe = load_universe()
    ev = load_evidence(refresh, universe)
    rows = build_rows(ev, universe)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with POOL_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["role"], r["name"])))
    print(f"wrote {POOL_CSV} ({len(rows)} rows, {sum(r['eligible'] for r in rows)} eligible)")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
