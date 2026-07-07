"""
Comparison results of a single column comparison across two tables.
Allows for inspection of the comparison results, including the passed/failed mask and the differing rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ColumnResult:
    """The outcome of comparing one column across the two tables.

    `passed` is the full per-key mask (True = match / within threshold). `left`
    and `right` hold display-ready values for the *differing* rows only — the
    comparator decides their representation.
    `measures` carries per-row numerics (e.g. percent_diff) surfaced per differing
    comparison in `differences_df`.
    """

    column: str
    method: str  # display label
    passed: pd.Series  # bool, indexed by key
    left: pd.Series  # differing rows only, indexed by key
    right: pd.Series
    measures: Optional[pd.DataFrame] = None  # per-key numerics, indexed by key

    @property
    def n_compared(self) -> int:
        return int(len(self.passed))

    @property
    def n_differences(self) -> int:
        return int((~self.passed).sum())

    @property
    def percent_diff(self) -> Optional[pd.Series]:
        """Per-row percent difference, if the comparator computed it."""
        if self.measures is not None and "percent_diff" in self.measures.columns:
            return self.measures["percent_diff"]
        return None


@dataclass
class ComparisonResult:
    """The full comparison: per-column results plus what didn't line up."""

    key: str
    columns: dict[str, ColumnResult]
    left_name: str = "table1"
    right_name: str = "table2"
    rows_only_left: list = field(default_factory=list)
    rows_only_right: list = field(default_factory=list)
    columns_only_left: list[str] = field(default_factory=list)
    columns_only_right: list[str] = field(default_factory=list)
    # configured columns that couldn't be resolved: column -> "left"|"right"|"both"
    missing_columns: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Deterministic gate CI should check on: every compared column matched,
        no rows missing on either side, and no configured column went missing.
        (Unconfigured column extras — `columns_only_*` — do not fail the gate.)"""
        no_col_diffs = all(
            column.n_differences == 0 for column in self.columns.values()
        )
        return (
            no_col_diffs
            and not self.rows_only_left
            and not self.rows_only_right
            and not self.missing_columns
        )

    def summary_df(self) -> pd.DataFrame:
        """One row per column: method and difference counts — a census of where
        differences are, not how big. Per-comparison detail lives in
        `differences_df`."""
        rows = [
            {
                "column": name,
                "method": column.method,
                "n_compared": column.n_compared,
                "n_differences": column.n_differences,
            }
            for name, column in self.columns.items()
        ]
        # explicit columns so an empty result (no columns compared) still has the
        # "column" index to set, rather than raising.
        cols = ["column", "method", "n_compared", "n_differences"]
        return pd.DataFrame(rows, columns=cols).set_index("column")

    def differences_df(self) -> pd.DataFrame:
        """Differing rows only, with (column, table) MultiIndex columns. Numeric
        columns also get a `percent_diff` sub-column, so each differing comparison
        shows its percent difference beside the two values."""
        frames = []
        for name, column in self.columns.items():
            if column.n_differences == 0:
                continue
            data = {
                (name, self.left_name): column.left,
                (name, self.right_name): column.right,
            }
            percent = column.percent_diff
            if percent is not None:
                # restrict to the differing rows so we don't reintroduce passing rows
                data[(name, "percent_diff")] = percent.reindex(column.left.index)
            frames.append(pd.DataFrame(data))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1)
        out.columns = pd.MultiIndex.from_tuples(out.columns, names=["column", "table"])
        out.index.name = self.key
        return out

    def to_dict(self) -> dict:
        """JSON-ready summary: counts and exclusives, no pandas. For CI asserts
        and passing results across process boundaries (e.g. bioforklift)."""
        return {
            "key": self.key,
            "passed": self.passed,
            "tables": {"left": self.left_name, "right": self.right_name},
            "exclusive_rows": {
                "left": list(self.rows_only_left),
                "right": list(self.rows_only_right),
            },
            "exclusive_columns": {
                "left": list(self.columns_only_left),
                "right": list(self.columns_only_right),
            },
            "missing_columns": dict(self.missing_columns),
            "columns": {
                name: {
                    "method": column.method,
                    "n_compared": column.n_compared,
                    "n_differences": column.n_differences,
                }
                for name, column in self.columns.items()
            },
        }

    def write(
        self,
        outdir: str,
        *,
        prefix: str = "theiavalidate",
        tsv: bool = True,
        html: bool = False,
        pdf: bool = False,
    ) -> Path:
        """Write artifacts. TSV here; html/pdf delegate to reporting.py (later)."""
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        if tsv:
            self.summary_df().to_csv(out / f"{prefix}_summary.tsv", sep="\t")
            diffs = self.differences_df()
            if not diffs.empty:
                diffs.to_csv(out / f"{prefix}_differences.tsv", sep="\t")
        if html or pdf:
            from theiavalidate import reporting  # lazy; built in a later phase

            reporting.render(self, out, prefix=prefix, html=html, pdf=pdf)
        return out
