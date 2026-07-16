"""Comparison methods: pure functions over two prepared (parsed+coerced) Series.

Each registered method returns `(pass_mask, measure)`:
  pass_mask  bool Series, True = match / within threshold (both-null counts as match)
  measure    optional numeric Series to surface (e.g. percent_diff, threshold), else None
This lets us know if the row passes the comparison and how far off it is or if it's an exact match.

`compare_column` is the orchestrator the validator calls: it checks method×type
"""
# Inner functions reference: https://www.geeksforgeeks.org/python/python-inner-functions/

from __future__ import annotations

from typing import Callable

import pandas as pd

from theiavalidate.config import KNOWN_METHODS, ColumnSpec, MethodSpec, TypeSpec
from theiavalidate.files import md5
from theiavalidate.results import ColumnResult

_NUMERIC = {"int", "float"}
_TEMPORAL = {"date", "datetime"}


def _finalize(mask: pd.Series, both_null: pd.Series) -> pd.Series:
    """Both-null will match; one-null will mismatch."""
    return mask.where(~both_null, True).fillna(False).astype(bool)


def _exact(left, right, *, threshold=None, type_spec=None):
    left_null, right_null = left.isna(), right.isna()
    both_null = left_null & right_null
    equal = (~left_null & ~right_null) & left.eq(right)
    return (both_null | equal).fillna(False).astype(bool), None


def _ignore(left, right, *, threshold=None, type_spec=None):
    return pd.Series(True, index=left.index), None


def _percent_diff(left, right, *, threshold, type_spec=None):
    """Percent diff is calculated as (left - right) / (abs(left) + abs(right))
    Threshold is expected as a fraction (e.g. 0.05 for 5%)"""
    left_val = pd.to_numeric(left, errors="coerce")
    right_val = pd.to_numeric(right, errors="coerce")
    denominator = (left_val.abs() + right_val.abs()) / 2
    percent = (left_val - right_val).abs() / denominator
    percent = percent.mask(
        left_val.eq(right_val), 0.0
    )  # equal values are 0% apart even if denominator is 0
    mask = _finalize(percent <= threshold, left_val.isna() & right_val.isna())
    # Return mask and percent diff series
    return mask, percent.rename("percent_diff")


def _range(left, right, *, threshold, type_spec=None):

    # Date range might be useful for clinical temporal comparisons
    if type_spec is not None and type_spec.base_type in _TEMPORAL:
        measure = _abs_days(left, right)
    else:
        left_val = pd.to_numeric(left, errors="coerce")
        right_val = pd.to_numeric(right, errors="coerce")
        measure = (left_val - right_val).abs()
    mask = _finalize(measure <= threshold, left.isna() & right.isna())
    # Return mask and range diff series
    return mask, measure.rename("range_diff")


def _abs_days(left: pd.Series, right: pd.Series) -> pd.Series:
    """Absolute day distance between two temporal Series (nulls -> NaN).

    Normalizes to datetime64 first so this works whether the column coerced to
    `datetime` (already datetime64) or `date` (Python date objects, object dtype).
    """
    left = pd.to_datetime(left, errors="coerce")
    right = pd.to_datetime(right, errors="coerce")
    return (left - right).abs().dt.total_seconds() / 86400.0  # 86400 seconds/day


def _file_exact(left, right, *, threshold=None, type_spec=None):
    """Files match when their contents are byte-identical (same md5). Each cell is
    a location (local path or gs:///s3:// URI). Both-null -> match; one-null ->
    mismatch."""

    def same(a, b) -> bool:
        a_null, b_null = pd.isna(a), pd.isna(b)
        if a_null and b_null:
            return True
        if a_null or b_null:
            return False
        return md5(str(a)) == md5(str(b))

    passed = {key: same(left[key], right[key]) for key in left.index}
    return pd.Series(passed, index=left.index, dtype=bool), None


def _file_set(*args, **kwargs):
    # This was in the orginal, wonder if we need this or another method
    raise NotImplementedError("file_set comparison is not implemented yet")


# registry of comparator functions, keyed by method name
COMPARATORS: dict[str, Callable] = {
    "exact": _exact,
    "ignore": _ignore,
    "percent_diff": _percent_diff,
    "range": _range,
    "file_exact": _file_exact,
    "file_set": _file_set,
}

# Runs at import to validate methods
assert set(COMPARATORS) == KNOWN_METHODS, (
    f"comparator registry {sorted(COMPARATORS)} != KNOWN_METHODS {sorted(KNOWN_METHODS)}"
)


def check_compatible(method: str, type_spec: TypeSpec, column: str) -> None:
    """Reject method×type combinations that make no sense"""
    if method in {"exact", "ignore"}:
        return  # exact works on any type
    if type_spec.is_container:
        raise ValueError(
            f"column {column!r}: method {method!r} needs a scalar type, got {type_spec}"
        )
    if method in {"file_exact", "file_set"}:
        return  # a scalar location string; content comparison handles the rest
    if method == "percent_diff" and type_spec.base_type not in _NUMERIC:
        raise ValueError(
            f"column {column!r}: percent_diff needs a numeric type, got {type_spec}"
        )
    if method == "range" and type_spec.base_type not in (_NUMERIC | _TEMPORAL):
        raise ValueError(
            f"column {column!r}: range needs a numeric or date type, got {type_spec}"
        )


def _label(methods: list[MethodSpec], combinator: str | None = None) -> str:
    def one(m: MethodSpec) -> str:
        return f"{m.method}({m.threshold})" if m.threshold is not None else m.method

    if len(methods) == 1:
        return one(methods[0])
    name = "all_of" if combinator == "all" else "any_of"
    return f"{name}(" + ", ".join(one(m) for m in methods) + ")"


def _format_diffs(left, right, idx, type_spec: TypeSpec):
    """Display values for differing rows. Sets show their symmetric difference."""
    if type_spec.container == "set":
        left_disp, right_disp = {}, {}
        for key in idx:
            left_val, right_val = left.get(key), right.get(key)
            if isinstance(left_val, set) and isinstance(right_val, set):
                left_disp[key] = ", ".join(sorted(map(str, left_val - right_val)))
                right_disp[key] = ", ".join(sorted(map(str, right_val - left_val)))
            else:
                left_disp[key], right_disp[key] = left_val, right_val
        return pd.Series(left_disp, dtype=object), pd.Series(right_disp, dtype=object)
    return left.loc[idx], right.loc[idx]


def compare_column(left: pd.Series, right: pd.Series, spec: ColumnSpec) -> ColumnResult:
    """Compare one prepared column across both tables into a ColumnResult.

    `left`/`right` are already parsed and coerced
    """
    for method_spec in spec.methods:
        check_compatible(method_spec.method, spec.type, spec.name)

    masks, measures = [], {}
    for method_spec in spec.methods:
        fn = COMPARATORS[method_spec.method]
        mask, measure = fn(
            left, right, threshold=method_spec.threshold, type_spec=spec.type
        )
        masks.append(mask)
        if measure is not None:
            measures[measure.name] = measure

    passed = masks[0]
    if spec.combinator == "all":
        for mask in masks[1:]:  # all_of: pass only if every branch passes
            passed = passed & mask
    else:
        for mask in masks[1:]:  # any_of (or single method): pass if any branch passes
            passed = passed | mask
    passed = passed.astype(bool)

    diff_index = passed.index[~passed]
    left_disparity, right_disparity = _format_diffs(left, right, diff_index, spec.type)
    return ColumnResult(
        column=spec.name,
        method=_label(spec.methods, spec.combinator),
        passed=passed,
        left=left_disparity,
        right=right_disparity,
        measures=pd.DataFrame(measures) if measures else None,
    )
