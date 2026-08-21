"""Parsing pipeline: parse_series, coerce_series, prepare_column."""

import numpy as np
import pandas as pd
import pytest

from theiavalidate.config import ColumnSpec, ParseSpec, TypeSpec
from theiavalidate.parsing import coerce_series, parse_series, prepare_column


class TestParseSeries:
    def test_none_returns_as_is(self):
        s = pd.Series(["a", "b"])
        assert parse_series(s, None) is s

    def test_delimiter_split_and_strip(self):
        s = pd.Series({"k": " a , b ,c "})
        out = parse_series(s, ParseSpec(method="delimiter", pattern=","))
        assert out["k"] == ["a", "b", "c"]

    def test_delimiter_drops_empty_fragments(self):
        s = pd.Series({"k": "a,,b,"})
        out = parse_series(s, ParseSpec(method="delimiter", pattern=","))
        assert out["k"] == ["a", "b"]

    def test_delimiter_preserves_null(self):
        s = pd.Series({"k": np.nan})
        out = parse_series(s, ParseSpec(method="delimiter", pattern=","))
        assert pd.isna(out["k"])

    def test_regex_extracts_named_group(self):
        s = pd.Series({"k": "coverage: 42x"})
        spec = ParseSpec(method="regex", pattern=r"(?P<v>\d+)x", field="v")
        assert parse_series(s, spec)["k"] == "42"

    def test_regex_non_match_is_null(self):
        s = pd.Series({"k": "no digits"})
        spec = ParseSpec(method="regex", pattern=r"(?P<v>\d+)", field="v")
        assert pd.isna(parse_series(s, spec)["k"])


class TestCoerceSeries:
    def test_scalar_coercion(self):
        out = coerce_series(pd.Series(["1", "2"]), TypeSpec.parse("int"), column="c")
        assert list(out) == [1, 2]

    def test_nulls_preserved(self):
        out = coerce_series(
            pd.Series([np.nan, "5"]), TypeSpec.parse("float"), column="c"
        )
        assert pd.isna(out.iloc[0]) and out.iloc[1] == 5.0

    def test_container_coercion(self):
        out = coerce_series(
            pd.Series([["a", "b"]]), TypeSpec.parse("set[str]"), column="c"
        )
        assert out.iloc[0] == {"a", "b"}

    def test_bad_value_error_names_column_and_key(self):
        s = pd.Series({"row1": "not-a-number"})
        with pytest.raises(ValueError, match=r"column 'c', key 'row1'"):
            coerce_series(s, TypeSpec.parse("float"), column="c")


class TestPrepareColumn:
    def test_delimiter_then_coerce(self):
        spec = ColumnSpec(name="c", type="set[str]", method="exact", delimiter=",")
        out = prepare_column(pd.Series({"k": "x,y"}), spec)
        assert out["k"] == {"x", "y"}

    def test_scalar_passthrough_then_coerce(self):
        spec = ColumnSpec(name="c", type="float", method="range", threshold=1.0)
        out = prepare_column(pd.Series({"k": "3.5"}), spec)
        assert out["k"] == 3.5
