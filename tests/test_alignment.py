"""Table alignment: row matching, column resolution, and exclusives."""

import pandas as pd
import pytest

from theiavalidate.alignment import align
from theiavalidate.config import Config


def _cfg(columns, **keys):
    return Config.from_dict({**keys, "columns": columns})


class TestAlign:
    def test_matches_rows_on_key_regardless_of_order(self):
        left = pd.DataFrame({"id": ["a", "b"], "x": ["1", "2"]})
        right = pd.DataFrame({"id": ["b", "a"], "x": ["2", "1"]})
        cfg = _cfg({"x": {"method": "exact"}}, key="id")
        aligned = align(left, right, cfg)
        assert list(aligned.left.index) == list(aligned.right.index)
        assert aligned.left.loc["a", "x"] == aligned.right.loc["a", "x"] == "1"

    def test_exclusive_rows(self):
        left = pd.DataFrame({"id": ["a", "b"], "x": ["1", "2"]})
        right = pd.DataFrame({"id": ["b", "c"], "x": ["2", "3"]})
        aligned = align(left, right, _cfg({"x": {"method": "exact"}}, key="id"))
        assert aligned.rows_only_left == ["a"]
        assert aligned.rows_only_right == ["c"]
        assert list(aligned.left.index) == ["b"]

    def test_column_resolved_via_mapping(self):
        left = pd.DataFrame({"id": ["a"], "old_name": ["1"]})
        right = pd.DataFrame({"id": ["a"], "new_name": ["1"]})
        cfg = _cfg(
            {"canonical": {"method": "exact", "mappings": ["old_name", "new_name"]}},
            key="id",
        )
        aligned = align(left, right, cfg)
        assert aligned.compared_columns == ["canonical"]
        assert aligned.left.loc["a", "canonical"] == "1"

    def test_missing_columns_classified(self):
        left = pd.DataFrame({"id": ["a"], "only_left": ["1"]})
        right = pd.DataFrame({"id": ["a"], "only_right": ["1"]})
        cfg = _cfg(
            {
                "only_left": {"method": "exact"},
                "only_right": {"method": "exact"},
                "nowhere": {"method": "exact"},
            },
            key="id",
        )
        aligned = align(left, right, cfg)
        assert aligned.missing_columns == {
            "only_left": "right",
            "only_right": "left",
            "nowhere": "both",
        }
        assert aligned.compared_columns == []

    def test_unconfigured_extra_columns_reported(self):
        left = pd.DataFrame({"id": ["a"], "x": ["1"], "extra_l": ["9"]})
        right = pd.DataFrame({"id": ["a"], "x": ["1"], "extra_r": ["8"]})
        aligned = align(left, right, _cfg({"x": {"method": "exact"}}, key="id"))
        assert aligned.columns_only_left == ["extra_l"]
        assert aligned.columns_only_right == ["extra_r"]

    def test_per_table_keys(self):
        left = pd.DataFrame({"lid": ["a"], "x": ["1"]})
        right = pd.DataFrame({"rid": ["a"], "x": ["1"]})
        cfg = _cfg({"x": {"method": "exact"}}, key1="lid", key2="rid")
        aligned = align(left, right, cfg)
        assert aligned.key == "lid"
        assert list(aligned.left.index) == ["a"]

    def test_missing_key_raises(self):
        left = pd.DataFrame({"id": ["a"], "x": ["1"]})
        right = pd.DataFrame({"nope": ["a"], "x": ["1"]})
        with pytest.raises(ValueError, match="key column 'id' not found in right"):
            align(left, right, _cfg({"x": {"method": "exact"}}, key="id"))

    def test_no_key_configured_raises(self):
        left = pd.DataFrame({"id": ["a"], "x": ["1"]})
        right = pd.DataFrame({"id": ["a"], "x": ["1"]})
        with pytest.raises(ValueError, match="no join key configured"):
            align(left, right, _cfg({"x": {"method": "exact"}}))

    def test_duplicate_keys_raise(self):
        left = pd.DataFrame({"id": ["a", "a"], "x": ["1", "2"]})
        right = pd.DataFrame({"id": ["a"], "x": ["1"]})
        with pytest.raises(ValueError, match="duplicate keys"):
            align(left, right, _cfg({"x": {"method": "exact"}}, key="id"))
