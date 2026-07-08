"""Turn raw table cells into comparable, coerced values.

This is the first two steps of the per-column pipeline (parse -> coerce ->
compare):

    parse_series   extract comparable value(s) from each cell
                   delimiter -> list[str] ; regex -> scalar
    coerce_series  coerce the parsed value(s) to the column's TypeSpec

Columns returned as a pandas Series for operations on the parsed values.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from pydantic import ValidationError

from theiavalidate.config import ColumnSpec, ParseSpec, TypeSpec


def _is_null(cell: object) -> bool:
    """Null check that is safe when the cell is already a list/set/tuple."""
    if isinstance(cell, (list, tuple, set)):
        return False
    return pd.isna(cell)


def _split(series: pd.Series, delimiter: str) -> pd.Series:
    """Split each cell on a literal delimiter into a stripped list of elements."""

    def split_cell(cell: object) -> object:
        if _is_null(cell):
            return cell
        parts = (p.strip() for p in str(cell).split(delimiter))
        return [p for p in parts if p != ""]

    return series.map(split_cell)


def _regex_extract(series: pd.Series, pattern: str, field: str) -> pd.Series:
    """Extract one named group from each cell; non-matches become null."""
    rx = re.compile(pattern)

    def extract_cell(cell: object) -> object:
        if _is_null(cell):
            return cell
        match = rx.search(str(cell))
        return match.group(field) if match else pd.NA

    return series.map(extract_cell)


def parse_series(series: pd.Series, spec: Optional[ParseSpec]) -> pd.Series:
    """Apply a parse spec to a column. `None` (no parse) returns the cells as-is."""
    if spec is None:
        return series
    if spec.method == "delimiter":
        return _split(series, spec.pattern)
    if spec.method == "regex":
        return _regex_extract(series, spec.pattern, spec.field)
    raise ValueError(f"unhandled parse method {spec.method!r}")  # pragma: no cover


def coerce_series(series: pd.Series, typespec: TypeSpec, *, column: str) -> pd.Series:
    """Coerce each parsed cell to the TypeSpec, preserving nulls.

    `column` is the configured config name

    Wraps pydantic's ValidationError with the offending column/key/value so a
    bad cell points at where it lives rather than surfacing a raw pydantic error.
    """
    out = []
    for key, cell in series.items():
        if _is_null(cell):
            out.append(cell)
            continue
        try:
            out.append(typespec.coerce(cell))
        except ValidationError as err:
            raise ValueError(
                f"column {column!r}, key {key!r}: cannot coerce {cell!r} to {typespec}"
            ) from err
    return pd.Series(out, index=series.index, name=series.name)


def prepare_column(series: pd.Series, spec: ColumnSpec) -> pd.Series:
    """Full pre-compare transform for one column: parse then coerce."""
    parsed = parse_series(series, spec.effective_parse)
    return coerce_series(parsed, spec.type, column=spec.name)
