"""Line the two tables up before comparison.


Order shouldn't matter as we match on the provided key.
The tables get aligned on the key, which
then resolved each configured column to it's source name.
Columns not found are recorded in `missing_columns`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from theiavalidate.config import ColumnSpec, Config


@dataclass
class Alignment:
    """Both tables reduced to shared rows and resolved columns, plus exclusives."""

    key: str
    left: pd.DataFrame
    right: pd.DataFrame
    rows_only_left: list = field(default_factory=list)
    rows_only_right: list = field(default_factory=list)
    columns_only_left: list[str] = field(default_factory=list)
    columns_only_right: list[str] = field(default_factory=list)
    missing_columns: dict[str, str] = field(
        default_factory=dict
    )  # column -> "left"|"right"|"both"

    @property
    def compared_columns(self) -> list[str]:
        return list(self.left.columns)


def _resolve(spec: ColumnSpec, columns) -> Optional[str]:
    """The source name for a configured column in a table as written else a mapping."""
    if spec.name in columns:
        return spec.name
    for alias in spec.mappings:
        if alias in columns:
            return alias
    return None


def align(left_df: pd.DataFrame, right_df: pd.DataFrame, config: Config) -> Alignment:
    left_key, right_key = config.left_key, config.right_key
    if left_key is None or right_key is None:
        # extra check here for safety
        raise ValueError(
            "no join key configured; set `key` (or `key1`+`key2`) in the config, "
            "or pass --key (or --key1/--key2) on the CLI"
        )
    for label, df, key in (("left", left_df, left_key), ("right", right_df, right_key)):
        if key not in df.columns:
            raise ValueError(f"key column {key!r} not found in {label} table")

    # Index on each table's own key, then give both indexes one canonical name so
    # the join and the reported key line up even when the source names differ.
    key = left_key
    left = left_df.set_index(left_key).rename_axis(key)
    right = right_df.set_index(right_key).rename_axis(key)
    for label, df in (("left", left), ("right", right)):
        if df.index.has_duplicates:
            dups = sorted(set(df.index[df.index.duplicated()]))
            raise ValueError(f"{label} table has duplicate keys for {key!r}: {dups}")

    # Rows: match on key; report the exclusives.
    shared = left.index.intersection(right.index)
    rows_only_left = list(left.index.difference(right.index))
    rows_only_right = list(right.index.difference(left.index))

    # Resolve each configured column in each table.
    left_out, right_out, missing = {}, {}, {}
    used_left, used_right = set(), set()
    for name, spec in config.columns.items():
        lname = _resolve(spec, left.columns)
        rname = _resolve(spec, right.columns)
        # A name that resolved on either side is "accounted for" — keep it out of
        # the unconfigured shape-diff report even if the pair is incomplete.
        if lname is not None:
            used_left.add(lname)
        if rname is not None:
            used_right.add(rname)
        if lname is None and rname is None:
            missing[name] = "both"
        elif lname is None:
            missing[name] = "left"
        elif rname is None:
            missing[name] = "right"
        else:
            left_out[name] = left.loc[shared, lname]
            right_out[name] = right.loc[shared, rname]

    # Raw column shape difference
    left_cols, right_cols = set(left.columns), set(right.columns)
    columns_only_left = sorted((left_cols - right_cols) - used_left)
    columns_only_right = sorted((right_cols - left_cols) - used_right)

    return Alignment(
        key=key,
        left=pd.DataFrame(left_out, index=shared),
        right=pd.DataFrame(right_out, index=shared),
        rows_only_left=rows_only_left,
        rows_only_right=rows_only_right,
        columns_only_left=columns_only_left,
        columns_only_right=columns_only_right,
        missing_columns=missing,
    )
