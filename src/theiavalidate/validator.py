"""Orchestrate a full comparison: align -> prepare -> compare -> ComparisonResult.

`Validator(config).compare(left_df, right_df)` is the class form; `compare_tables`
is the one-call shortcut.

Per-column pipeline for each column that resolved in both tables:
    mark na_values -> parse -> coerce -> compare
"""

from __future__ import annotations

import pandas as pd

from theiavalidate.alignment import align
from theiavalidate.comparators import compare_column
from theiavalidate.config import ColumnSpec, Config
from theiavalidate.parsing import prepare_column
from theiavalidate.results import ColumnResult, ComparisonResult


class Validator:
    """Holds a Config and compares table pairs against it."""

    def __init__(self, config: Config):
        self.config = config

    def compare(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        *,
        left_name: str = "table1",
        right_name: str = "table2",
    ) -> ComparisonResult:
        # Align the tables to be compared
        aligned = align(left_df, right_df, self.config)

        columns: dict[str, ColumnResult] = {}
        for name in aligned.compared_columns:
            spec = self.config.columns[name]
            na = self._na_values(spec)
            left = prepare_column(self._mark_na(aligned.left[name], na), spec)
            right = prepare_column(self._mark_na(aligned.right[name], na), spec)
            columns[name] = compare_column(left, right, spec)

        return ComparisonResult(
            key=aligned.key,
            columns=columns,
            left_name=left_name,
            right_name=right_name,
            rows_only_left=aligned.rows_only_left,
            rows_only_right=aligned.rows_only_right,
            columns_only_left=aligned.columns_only_left,
            columns_only_right=aligned.columns_only_right,
            missing_columns=aligned.missing_columns,
        )

    # Have it live here so we can use it in `align` as well
    def _na_values(self, spec: ColumnSpec) -> set[str]:
        """Global na_values plus this column's optional extensions."""
        values = set(self.config.na_values)
        if spec.na_values:
            values.update(spec.na_values)
        return values

    @staticmethod
    def _mark_na(series: pd.Series, na_values: set[str]) -> pd.Series:
        """Replace whole-cell na sentinels with NaN before parsing."""
        return series.mask(series.isin(na_values))


def compare_tables(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    config: Config,
    *,
    left_name: str = "table1",
    right_name: str = "table2",
) -> ComparisonResult:
    """One-call shortcut for `Validator(config).compare(...)`."""
    return Validator(config).compare(
        left_df, right_df, left_name=left_name, right_name=right_name
    )
