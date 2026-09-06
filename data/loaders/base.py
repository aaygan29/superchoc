"""Shared types and source registry for dataset loaders."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


class DataNotAvailable(RuntimeError):
    """Raised when a loader's underlying source has not been cached locally yet."""


@dataclass
class LoaderResult:
    """Uniform return type for every loader once data is wired in."""

    name: str
    n_rows: int
    columns: list
    source_url: str
    local_path: str
    notes: str = ""


@dataclass
class Source:
    key: str
    name: str
    url: str
    layer: str            # "A" percept, "B" receptor, "C" affinity (see GROUND_TRUTH.md)
    expected_columns: list = field(default_factory=list)
    license_note: str = ""


# Registry mirrors docs/GROUND_TRUTH.md. URLs are the canonical landing pages.
SOURCES = {
    "pyrfume": Source(
        "pyrfume", "Pyrfume archive",
        "https://www.nature.com/articles/s41597-024-04051-z", "A",
        ["cid", "smiles", "descriptors"],
        "Aggregated open olfactory data; see pyrfume-data on GitHub.",
    ),
    "leffingwell_goodscents": Source(
        "leffingwell_goodscents", "GoodScents + Leffingwell odor labels",
        "https://doi.org/10.5281/zenodo.4085097", "A",
        ["smiles", "odor_labels"],
        "~5030 expert-labeled molecules, multi-label.",
    ),
    "dream": Source(
        "dream", "DREAM Olfaction Prediction Challenge (Keller et al. 2017)",
        "https://doi.org/10.1093/gigascience/gix127", "A",
        ["cid", "smiles", "intensity", "pleasantness", "per_subject_ratings"],
        "480 molecules, continuous + personalized ratings; a held-out benchmark.",
    ),
    "chembl": Source(
        "chembl", "ChEMBL bioactivities (GPCR targets)",
        "https://www.ebi.ac.uk/chembl/", "C",
        ["molecule_chembl_id", "smiles", "target_chembl_id", "standard_type", "standard_value"],
        "Filter to GPCR targets, confidence score 9 for highest quality.",
    ),
    "gpcrdb": Source(
        "gpcrdb", "GPCRdb (structures, ligands, incl. odorant receptors as of 2025)",
        "https://gpcrdb.org/", "B",
        ["receptor", "uniprot", "structure_model", "ligand_associations"],
        "Bridges olfactory (OR) and therapeutic GPCR sides of the project.",
    ),
}


def default_cache_dir() -> str:
    return os.environ.get("SUPERCHOC_DATA", os.path.join(os.path.dirname(__file__), "..", "raw"))


def describe_sources() -> str:
    lines = ["key | layer | name | url"]
    for s in SOURCES.values():
        lines.append(f"{s.key} | {s.layer} | {s.name} | {s.url}")
    return "\n".join(lines)


def _require_cached(source_key: str, filename: str) -> str:
    """Return the local path if the cached file exists, else raise DataNotAvailable."""
    src = SOURCES[source_key]
    path = os.path.abspath(os.path.join(default_cache_dir(), filename))
    if not os.path.exists(path):
        raise DataNotAvailable(
            f"{src.name} not cached at {path}. Download from {src.url} and place it there, "
            f"or set $SUPERCHOC_DATA. This loader does not fabricate data."
        )
    return path
