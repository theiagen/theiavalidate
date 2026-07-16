"""Comparison methods, method×type compatibility, labels, and combinators."""

import numpy as np
import pandas as pd
import pytest

from theiavalidate.comparators import (
    _label,
    check_compatible,
    compare_column,
)
from theiavalidate.config import ColumnSpec, MethodSpec, TypeSpec
from theiavalidate.parsing import prepare_column


def _prep(spec, left, right):
    return prepare_column(pd.Series(left), spec), prepare_column(pd.Series(right), spec)


class TestExact:
    def test_scalar_equality(self):
        spec = ColumnSpec(name="c", type="str", method="exact")
        left, right = _prep(spec, {"a": "x", "b": "y"}, {"a": "x", "b": "z"})
        assert dict(compare_column(left, right, spec).passed) == {"a": True, "b": False}

    def test_set_equality_is_order_independent(self):
        spec = ColumnSpec(name="c", type="set[str]", method="exact", delimiter=",")
        left, right = _prep(spec, {"a": "x,y", "b": "m,n"}, {"a": "y,x", "b": "m,z"})
        assert dict(compare_column(left, right, spec).passed) == {"a": True, "b": False}

    def test_both_null_matches_one_null_mismatches(self):
        spec = ColumnSpec(name="c", type="str", method="exact")
        left, right = _prep(
            spec, {"a": np.nan, "b": np.nan}, {"a": np.nan, "b": "z"}
        )
        assert dict(compare_column(left, right, spec).passed) == {"a": True, "b": False}

    def test_set_diff_rendered_as_plain_string(self):
        # both sides present: show only the differing elements, no {...} repr.
        spec = ColumnSpec(name="c", type="set[str]", method="exact", delimiter=",")
        left, right = _prep(spec, {"a": "x,y,z"}, {"a": "x,y,w"})
        result = compare_column(left, right, spec)
        assert result.left["a"] == "z"
        assert result.right["a"] == "w"

    def test_set_vs_null_rendered_as_plain_string(self):
        # one side null: show the present set in full, still without {...}.
        spec = ColumnSpec(name="c", type="set[str]", method="exact", delimiter=",")
        left, right = _prep(spec, {"a": "hmrM,tetB"}, {"a": np.nan})
        result = compare_column(left, right, spec)
        assert result.left["a"] == "hmrM, tetB"
        assert pd.isna(result.right["a"])


class TestIgnore:
    def test_always_passes(self):
        spec = ColumnSpec(name="c", type="str", method="ignore")
        left, right = _prep(spec, {"a": "x", "b": np.nan}, {"a": "DIFFERENT", "b": "z"})
        result = compare_column(left, right, spec)
        assert result.passed.all()
        assert result.n_differences == 0


class TestPercentDiff:
    def test_within_threshold(self):
        spec = ColumnSpec(name="c", type="float", method="percent_diff", threshold=0.05)
        left, right = _prep(spec, {"a": 100.0, "b": 100.0}, {"a": 102.0, "b": 120.0})
        assert dict(compare_column(left, right, spec).passed) == {"a": True, "b": False}

    def test_equal_values_zero_percent_even_if_zero(self):
        spec = ColumnSpec(name="c", type="float", method="percent_diff", threshold=0.01)
        left, right = _prep(spec, {"a": 0.0}, {"a": 0.0})
        assert dict(compare_column(left, right, spec).passed) == {"a": True}

    def test_measure_surfaced(self):
        spec = ColumnSpec(name="c", type="float", method="percent_diff", threshold=0.0)
        left, right = _prep(spec, {"a": 100.0}, {"a": 110.0})
        result = compare_column(left, right, spec)
        assert result.percent_diff is not None
        assert result.percent_diff["a"] == pytest.approx(10 / 105)


class TestRange:
    def test_numeric_within_delta(self):
        spec = ColumnSpec(name="c", type="float", method="range", threshold=1.0)
        left, right = _prep(spec, {"a": 100.0, "b": 100.0}, {"a": 100.5, "b": 102.0})
        assert dict(compare_column(left, right, spec).passed) == {"a": True, "b": False}

    @pytest.mark.parametrize("temporal_type", ["datetime", "date"])
    def test_temporal_within_days(self, temporal_type):
        spec = ColumnSpec(name="c", type=temporal_type, method="range", threshold=2)
        left, right = _prep(
            spec,
            {"a": "2021-01-01", "b": "2021-01-10"},
            {"a": "2021-01-02", "b": "2021-01-20"},
        )
        assert dict(compare_column(left, right, spec).passed) == {"a": True, "b": False}

    def test_temporal_both_null_matches_one_null_mismatches(self):
        spec = ColumnSpec(name="c", type="date", method="range", threshold=2)
        left, right = _prep(
            spec,
            {"a": np.nan, "b": np.nan},
            {"a": np.nan, "b": "2021-01-01"},
        )
        assert dict(compare_column(left, right, spec).passed) == {"a": True, "b": False}


class TestFileExact:
    def test_matches_and_mismatches(self, tmp_path):
        same_a = tmp_path / "same_a"
        same_b = tmp_path / "same_b"
        diff = tmp_path / "diff"
        same_a.write_text("identical")
        same_b.write_text("identical")
        diff.write_text("other")
        spec = ColumnSpec(name="c", type="str", method="file_exact")
        left = pd.Series({"a": str(same_a), "b": str(same_a), "c": str(same_a)})
        right = pd.Series({"a": str(same_b), "b": str(diff), "c": str(same_a)})
        result = compare_column(left, right, spec)
        assert dict(result.passed) == {"a": True, "b": False, "c": True}

    def test_both_null_matches_one_null_mismatches(self, tmp_path):
        f = tmp_path / "f"
        f.write_text("x")
        spec = ColumnSpec(name="c", type="str", method="file_exact")
        left = pd.Series({"a": np.nan, "b": np.nan}, dtype=object)
        right = pd.Series({"a": np.nan, "b": str(f)}, dtype=object)
        assert dict(compare_column(left, right, spec).passed) == {
            "a": True,
            "b": False,
        }


class TestCheckCompatible:
    def test_exact_and_ignore_any_type(self):
        check_compatible("exact", TypeSpec.parse("set[str]"), "c")
        check_compatible("ignore", TypeSpec.parse("list[int]"), "c")

    def test_numeric_method_rejects_container(self):
        with pytest.raises(ValueError, match="scalar type"):
            check_compatible("range", TypeSpec.parse("set[float]"), "c")

    def test_percent_diff_needs_numeric(self):
        with pytest.raises(ValueError, match="numeric type"):
            check_compatible("percent_diff", TypeSpec.parse("str"), "c")

    def test_range_allows_numeric_and_date(self):
        check_compatible("range", TypeSpec.parse("float"), "c")
        check_compatible("range", TypeSpec.parse("date"), "c")

    def test_range_rejects_non_numeric_non_date(self):
        with pytest.raises(ValueError, match="numeric or date"):
            check_compatible("range", TypeSpec.parse("str"), "c")


class TestLabel:
    def test_single_method(self):
        assert _label([MethodSpec(method="exact")]) == "exact"

    def test_single_method_with_threshold(self):
        assert _label([MethodSpec(method="range", threshold=5.0)]) == "range(5.0)"

    def test_any_of_default(self):
        methods = [
            MethodSpec(method="percent_diff", threshold=0.01),
            MethodSpec(method="range", threshold=1.0),
        ]
        assert _label(methods, "any") == "any_of(percent_diff(0.01), range(1.0))"

    def test_all_of(self):
        methods = [
            MethodSpec(method="percent_diff", threshold=0.01),
            MethodSpec(method="range", threshold=1.0),
        ]
        assert _label(methods, "all") == "all_of(percent_diff(0.01), range(1.0))"


class TestCombinators:
    # "d": abs delta large (50) but pct tiny (0.5%) -> range fails, percent_diff passes
    # "e": abs delta tiny (0.5) but pct large (50%) -> range passes, percent_diff fails
    BRANCHES = [
        {"method": "percent_diff", "threshold": 0.01},
        {"method": "range", "threshold": 1.0},
    ]
    LEFT = {"d": 10000.0, "e": 1.0}
    RIGHT = {"d": 10050.0, "e": 1.5}

    def _passed(self, kind):
        spec = ColumnSpec(name="c", type="float", **{kind: self.BRANCHES})
        left, right = _prep(spec, self.LEFT, self.RIGHT)
        return dict(compare_column(left, right, spec).passed)

    def test_any_of_passes_when_either_branch_passes(self):
        assert self._passed("any_of") == {"d": True, "e": True}

    def test_all_of_requires_every_branch(self):
        assert self._passed("all_of") == {"d": False, "e": False}

    def test_labels_reflect_combinator(self):
        for kind, prefix in (("any_of", "any_of("), ("all_of", "all_of(")):
            spec = ColumnSpec(name="c", type="float", **{kind: self.BRANCHES})
            left, right = _prep(spec, self.LEFT, self.RIGHT)
            assert compare_column(left, right, spec).method.startswith(prefix)
