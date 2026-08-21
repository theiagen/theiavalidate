"""
Simple utilities for resolving named workflow presets to local YAML paths or PHB repo URLs.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

# Where the presets live in the public_health_bioinformatics repo.
_REPO = "theiagen/public_health_bioinformatics"
_REMOTE_SUBDIR = "tests/config/theiavalidate"
_RAW_BASE = "https://raw.githubusercontent.com"

DEFAULT_REF = "main"


def _normalize_name(name: str) -> str:
    """Accept a bare workflow name or a `*.yaml`/`*.yml` filename."""
    for suffix in (".yaml", ".yml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _cache_dir(ref: str) -> Path:
    """Per-ref cache dir, honoring XDG_CACHE_HOME."""
    root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(root) / "theiavalidate" / "presets" / ref


def remote_url(name: str, ref: str = DEFAULT_REF) -> str:
    """The raw.githubusercontent.com URL for a preset at a given ref."""
    return f"{_RAW_BASE}/{_REPO}/{ref}/{_REMOTE_SUBDIR}/{name}.yaml"


def _download(url: str, name: str) -> str:
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        if err.code == 404:
            raise ValueError(
                f"unknown preset {name!r}: {url} returned 404. Check the name, "
                "or pass --preset-dir to use a local preset directory."
            ) from err
        raise ValueError(f"failed to download preset {name!r} from {url}: {err}") from err
    except urllib.error.URLError as err:
        raise ValueError(
            f"could not reach {url} to download preset {name!r}: {err.reason}"
        ) from err


def resolve_preset(
    name: str,
    *,
    preset_dir: str | os.PathLike[str] | None = None,
    ref: str = DEFAULT_REF,
    refresh: bool = False,
) -> Path:
    """Return a local path to the preset YAML for `name`.

    With `preset_dir`, read it from there. Otherwise download from the theiagen
    repo at `ref` and cache it locally.
    """
    name = _normalize_name(name)

    if preset_dir is not None:
        path = Path(preset_dir) / f"{name}.yaml"
        if not path.is_file():
            raise ValueError(
                f"preset {name!r} not found in {os.fspath(preset_dir)!r} "
                f"(expected {path})"
            )
        return path

    cached = _cache_dir(ref) / f"{name}.yaml"
    if cached.is_file() and not refresh:
        return cached

    text = _download(remote_url(name, ref), name)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(text)
    return cached
