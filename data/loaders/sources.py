"""Per-source loader entry points.

Each returns a LoaderResult once the source is cached locally (see base._require_cached).
Until then it raises DataNotAvailable with instructions. Wiring a real loader means:
parse the cached file, validate columns against Source.expected_columns, return metadata.
"""

from __future__ import annotations

from .base import LoaderResult, SOURCES, _require_cached


def load_pyrfume() -> LoaderResult:
    path = _require_cached("pyrfume", "pyrfume.parquet")
    raise NotImplementedError("Parse pyrfume archive at %s into LoaderResult." % path)


def load_leffingwell_goodscents() -> LoaderResult:
    path = _require_cached("leffingwell_goodscents", "leffingwell_goodscents.csv")
    raise NotImplementedError("Parse odor-label CSV at %s." % path)


def load_dream() -> LoaderResult:
    path = _require_cached("dream", "dream_olfaction.csv")
    raise NotImplementedError("Parse DREAM challenge data at %s." % path)


def load_chembl_gpcr() -> LoaderResult:
    path = _require_cached("chembl", "chembl_gpcr_bioactivity.csv")
    raise NotImplementedError("Parse ChEMBL GPCR bioactivity export at %s." % path)


def load_gpcrdb() -> LoaderResult:
    path = _require_cached("gpcrdb", "gpcrdb.json")
    raise NotImplementedError("Parse GPCRdb export at %s." % path)


LOADERS = {
    "pyrfume": load_pyrfume,
    "leffingwell_goodscents": load_leffingwell_goodscents,
    "dream": load_dream,
    "chembl": load_chembl_gpcr,
    "gpcrdb": load_gpcrdb,
}

assert set(LOADERS) == set(SOURCES), "every source needs a loader"
