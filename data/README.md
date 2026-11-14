# data/

| folder | contents | licence / provenance |
|---|---|---|
| `hedonics/` | Ma et al. 2021 binary-mixture pleasantness (test set), plus Keller 2016, Ravia 2020, Arshamian 2022, Weiss 2012 files fetched during the search. `HEDONICS.md` records what each contains and the mixture-liking rules found in the literature | Ma 2021: the Data in Brief article is CC BY 4.0 (Crossref); the dataset record (Portail Data INRAE, doi 10.15454/51OVY6) states no licence in its DataCite metadata, so reuse terms should be confirmed with the authors before publication. Keller 2016, Ravia 2020, Arshamian 2022, Weiss 2012: copies of the pyrfume archive files; confirm each source's terms before redistribution, or fetch them through pyrfume instead |
| `sensomics/` | cocoa and dark-chocolate key odorants: concentrations, odor thresholds, OAVs, with the table and page each number came from (`SOURCES.md`) | values extracted from open-access papers; short quotes only |
| `candidates/` | the prospective candidate pool, its regulatory evidence, and the frozen predictions | built by `methods/chocolate_anchor/` |
| `raw/m2or_*.csv` | M2OR subset used by the exploratory receptor model | M2OR (Lalis et al. 2024) |

No value here is invented. Where a value could not be sourced it is blank, and the gap is
listed in that folder's notes.
