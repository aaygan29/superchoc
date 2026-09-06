"""Dataset loaders for the superchoc ground-truth corpus.

Each loader targets one public source described in docs/GROUND_TRUTH.md. These are
SCAFFOLDING: they define a stable interface and document the source, cache path, and
expected schema. They do not ship the data (licenses/size), and they do not fabricate
it. A loader raises DataNotAvailable until the real source is wired in and cached
locally, so nothing downstream silently trains on invented numbers.
"""

from .base import DataNotAvailable, LoaderResult, describe_sources

__all__ = ["DataNotAvailable", "LoaderResult", "describe_sources"]
