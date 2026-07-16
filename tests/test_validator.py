"""End-to-end validator: align -> prepare -> compare, plus na handling."""

import pandas as pd
import pytest

from theiavalidate.config import Config
from theiavalidate.validator import Validator, compare_tables


def _cfg(columns, **keys):
    return Config.from_dict({**keys, "columns": columns})


class TestCompareTables:
    def test_clean_comparison_passes(self):
        left = pd.DataFrame({"id": ["a", "b"], "taxon": ["E.coli", "S.aureus"]})
        right = pd.DataFrame({"id": ["b", "a"], "taxon": ["S.aureus", "E.coli"]})
        cfg = _cfg({"taxon": {"method": "exact"}}, key="id")
        result = compare_tables(left, right, cfg)
        assert result.passed
        assert result.columns["taxon"].n_differences == 0

    def test_difference_detected(self):
        left = pd.DataFrame({"id": ["a"], "len": ["100"]})
        right = pd.DataFrame({"id": ["a"], "len": ["200"]})
        cfg = _cfg(
            {"len": {"type": "float", "method": "percent_diff", "threshold": 0.01}},
            key="id",
        )
        result = compare_tables(left, right, cfg)
        assert not result.passed
        assert result.columns["len"].n_differences == 1

    def test_table_names_flow_through(self):
        left = pd.DataFrame({"id": ["a"], "x": ["1"]})
        right = pd.DataFrame({"id": ["a"], "x": ["1"]})
        result = compare_tables(
            left, right, _cfg({"x": {"method": "exact"}}, key="id"),
            left_name="run1.tsv", right_name="run2.tsv",
        )
        assert result.left_name == "run1.tsv" and result.right_name == "run2.tsv"


class TestNaHandling:
    def test_global_na_values_become_null(self):
        # "NA" is a default sentinel -> both treated null -> match
        left = pd.DataFrame({"id": ["a"], "x": ["NA"]})
        right = pd.DataFrame({"id": ["a"], "x": [""]})
        result = compare_tables(left, right, _cfg({"x": {"method": "exact"}}, key="id"))
        assert result.passed

    def test_one_null_one_value_mismatches(self):
        left = pd.DataFrame({"id": ["a"], "x": ["NA"]})
        right = pd.DataFrame({"id": ["a"], "x": ["real"]})
        result = compare_tables(left, right, _cfg({"x": {"method": "exact"}}, key="id"))
        assert not result.passed

    def test_per_column_na_extension(self):
        left = pd.DataFrame({"id": ["a"], "x": ["MISSING"]})
        right = pd.DataFrame({"id": ["a"], "x": [""]})
        cfg = _cfg(
            {"x": {"method": "exact", "na_values": ["MISSING"]}}, key="id"
        )
        result = compare_tables(left, right, cfg)
        assert result.passed


class TestEndToEnd:
    """One realistic comparison exercising the whole pipeline: key-join
    alignment, per-column prepare+compare, and the resulting artifacts. This is
    the library's core promise, so it asserts actual content, not just the gate."""

    def _result(self):
        left = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "taxon": ["E.coli", "S.aureus", "X"],
                "len": ["100", "200", "300"],
            }
        )
        right = pd.DataFrame(
            {
                "id": ["b", "c", "d"],  # 'a' only left, 'd' only right
                "taxon": ["S.aureus", "Y", "Z"],
                "len": ["100", "305", "400"],
            }
        )
        cfg = _cfg(
            {
                "taxon": {"method": "exact"},
                "len": {"type": "float", "method": "percent_diff", "threshold": 0.01},
                "absent": {"method": "exact"},  # configured but in neither table
            },
            key="id",
        )
        return compare_tables(left, right, cfg)

    def test_only_shared_rows_are_compared(self):
        # 'a'/'d' are exclusive; only the shared keys b, c get compared.
        result = self._result()
        assert list(result.columns["taxon"].passed.index) == ["b", "c"]
        assert result.columns["taxon"].n_compared == 2

    def test_exclusive_rows_recorded(self):
        result = self._result()
        assert result.rows_only_left == ["a"]
        assert result.rows_only_right == ["d"]

    def test_configured_column_missing_from_both(self):
        assert self._result().missing_columns == {"absent": "both"}

    def test_differences_carry_real_values_and_percent_diff(self):
        diffs = self._result().differences_df()
        # taxon differs only at c (X vs Y); b matched so is absent.
        assert diffs.loc["c", ("taxon", "table1")] == "X"
        assert diffs.loc["c", ("taxon", "table2")] == "Y"
        assert pd.isna(diffs.loc["b", ("taxon", "table1")])
        # len differs at both b and c, with a surfaced percent_diff.
        assert diffs.loc["b", ("len", "percent_diff")] == pytest.approx(100 / 150)

    def test_gate_fails_on_any_of_diffs_exclusives_or_missing(self):
        assert not self._result().passed

    def test_to_dict_round_trips_the_whole_picture(self):
        d = self._result().to_dict()
        assert d["passed"] is False
        assert d["exclusive_rows"] == {"left": ["a"], "right": ["d"]}
        assert d["missing_columns"] == {"absent": "both"}
        assert d["columns"]["len"]["n_differences"] == 2
        assert d["columns"]["taxon"]["n_differences"] == 1


class TestValidatorClass:
    def test_reuses_config_across_calls(self):
        cfg = _cfg({"x": {"method": "exact"}}, key="id")
        validator = Validator(cfg)
        a = pd.DataFrame({"id": ["a"], "x": ["1"]})
        b = pd.DataFrame({"id": ["a"], "x": ["1"]})
        c = pd.DataFrame({"id": ["a"], "x": ["2"]})
        assert validator.compare(a, b).passed
        assert not validator.compare(a, c).passed
