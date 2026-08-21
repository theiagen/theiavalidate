"""Resolve a file location to a content hash for file based comparisons."""

from __future__ import annotations

import hashlib
from functools import lru_cache

import fsspec

_CHUNK = 1 << 20  # 1 MiB seems reasonable

# NOTE: fsspec handles local paths and remote URIs uniformly
# (https://filesystem-spec.readthedocs.io/en/latest/features.html)


@lru_cache(maxsize=None)
def md5(uri: str) -> str:
    """MD5 hex digest of the file at `uri`, local or remote."""
    digest = hashlib.md5()
    with fsspec.open(uri, "rb") as fh:
        # Read in bytes for hashlib
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
