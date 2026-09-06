# data/

Scaffold for the ground-truth corpus that anchors the three success criteria. See
[../docs/GROUND_TRUTH.md](../docs/GROUND_TRUTH.md) for what each source certifies.

## Layout

```
data/
  raw/         # cached source files (gitignored; you download these)
  processed/   # cleaned, split-ready tables you build
  loaders/     # import-safe loaders with a uniform interface
```

## Principle

Loaders do **not** ship data and do **not** fabricate it. Each loader targets one public
source, documents the URL and expected columns, and raises `DataNotAvailable` until you
download the source into `raw/` (or set `$SUPERCHOC_DATA`). This keeps invented numbers
out of anything downstream.

## Sources (mirror of GROUND_TRUTH.md)

| key | layer | source |
|---|---|---|
| `pyrfume` | A (percept) | Pyrfume archive |
| `leffingwell_goodscents` | A (percept) | GoodScents + Leffingwell odor labels |
| `dream` | A (percept) | DREAM Olfaction Challenge (Keller et al. 2017) |
| `gpcrdb` | B (receptor) | GPCRdb (incl. odorant receptors, 2025) |
| `chembl` | C (affinity) | ChEMBL GPCR bioactivities |

Layers: A = molecule to percept (Criterion 1), B = molecule to receptor activation
(bridge), C = ligand to GPCR affinity (Criterion 3).

## Usage

```python
from loaders import describe_sources
from loaders.sources import LOADERS

print(describe_sources())          # list sources and URLs
LOADERS["dream"]()                 # raises DataNotAvailable until cached
```

Wiring a real loader: download the source into `raw/`, then implement the parse step
(the loader currently raises `NotImplementedError` past the cache check) to return a
`LoaderResult` validated against the source's `expected_columns`.
