"""ComparisonResult/ColumnResult: gate, summaries, diffs, to_dict, write."""

import pandas as pd

from theiavalidate.results import ColumnResult, ComparisonResult


def _col(name, passed, left=None, right=None, measures=None):
    passed = pd.Series(passed)
    diff_idx = passed.index[~passed]
    return ColumnResult(
        column=name,
        method="exact",
        passed=passed,
        left=pd.Series(left or {}, dtype=object).reindex(diff_idx),
        right=pd.Series(right or {}, dtype=object).reindex(diff_idx),
        measures=measures,
    )


class TestColumnResult:
    def test_counts(self):
        col = _col("c", {"a": True, "b": False, "c": False})
        assert col.n_compared == 3
        assert col.n_differences == 2

    def test_percent_diff_present(self):
        measures = pd.DataFrame({"percent_diff": pd.Series({"a": 0.1})})
        col = _col("c", {"a": False}, left={"a": "1"}, right={"a": "2"}, measures=measures)
        assert col.percent_diff["a"] == 0.1

    def test_percent_diff_absent(self):
        assert _col("c", {"a": True}).percent_diff is None


class TestComparisonResultGate:
    def _result(self, **kw):
        base = dict(key="id", columns={"c": _col("c", {"a": True})})
        base.update(kw)
        return ComparisonResult(**base)

    def test_passes_when_clean(self):
        assert self._result().passed

    def test_fails_on_column_diff(self):
        r = self._result(columns={"c": _col("c", {"a": False}, {"a": "1"}, {"a": "2"})})
        assert not r.passed

    def test_fails_on_exclusive_rows(self):
        assert not self._result(rows_only_left=["z"]).passed
        assert not self._result(rows_only_right=["z"]).passed

    def test_fails_on_missing_columns(self):
        assert not self._result(missing_columns={"c": "left"}).passed

    def test_unconfigured_extras_do_not_fail_gate(self):
        assert self._result(columns_only_left=["extra"]).passed


class TestFrames:
    def _result(self):
        cols = {
            "ok": _col("ok", {"a": True, "b": True}),
            "bad": _col("bad", {"a": True, "b": False}, {"b": "1"}, {"b": "2"}),
        }
        return ComparisonResult(key="id", columns=cols, left_name="L", right_name="R")

    def test_summary_df(self):
        summary = self._result().summary_df()
        assert list(summary.index) == ["ok", "bad"]
        assert summary.loc["bad", "n_differences"] == 1
        assert summary.loc["ok", "n_differences"] == 0

    def test_summary_df_empty(self):
        summary = ComparisonResult(key="id", columns={}).summary_df()
        assert summary.index.name == "column"
        assert summary.empty

    def test_differences_df_only_differing(self):
        diffs = self._result().differences_df()
        assert ("bad", "L") in diffs.columns
        assert ("ok", "L") not in diffs.columns
        assert list(diffs.index) == ["b"]

    def test_differences_df_empty_when_clean(self):
        clean = ComparisonResult(key="id", columns={"ok": _col("ok", {"a": True})})
        assert clean.differences_df().empty

    def test_differences_long_df(self):
        long = self._result().differences_long_df()
        assert list(long["column"]) == ["bad"]
        assert long.iloc[0]["L"] == "1" and long.iloc[0]["R"] == "2"
        # percent_diff dropped when no column produced it
        assert "percent_diff" not in long.columns


class TestToDict:
    def test_shape(self):
        cols = {"bad": _col("bad", {"a": False}, {"a": "1"}, {"a": "2"})}
        result = ComparisonResult(
            key="id", columns=cols, rows_only_left=["z"], missing_columns={"m": "both"}
        )
        d = result.to_dict()
        assert d["key"] == "id"
        assert d["passed"] is False
        assert d["exclusive_rows"]["left"] == ["z"]
        assert d["missing_columns"] == {"m": "both"}
        assert d["columns"]["bad"]["n_differences"] == 1


class TestWrite:
    def test_writes_summary_and_differences_tsv(self, tmp_path):
        cols = {"bad": _col("bad", {"a": False}, {"a": "1"}, {"a": "2"})}
        result = ComparisonResult(key="id", columns=cols)
        result.write(str(tmp_path), prefix="t", tsv=True, html=False)
        assert (tmp_path / "t_summary.tsv").exists()
        assert (tmp_path / "t_differences.tsv").exists()

    def test_no_differences_file_when_clean(self, tmp_path):
        result = ComparisonResult(key="id", columns={"ok": _col("ok", {"a": True})})
        result.write(str(tmp_path), prefix="t", tsv=True, html=False)
        assert (tmp_path / "t_summary.tsv").exists()
        assert not (tmp_path / "t_differences.tsv").exists()
