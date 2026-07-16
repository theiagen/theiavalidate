"""Config models: TypeSpec, ParseSpec, MethodSpec, ColumnSpec, Config."""

import datetime as dt

import pytest
from pydantic import ValidationError

from theiavalidate.config import (
    ColumnSpec,
    Config,
    MethodSpec,
    ParseSpec,
    TypeSpec,
)


class TestTypeSpec:
    @pytest.mark.parametrize(
        "raw,base,container",
        [
            ("str", "str", None),
            ("float", "float", None),
            ("set[str]", "str", "set"),
            ("list[float]", "float", "list"),
            ("  int  ", "int", None),
        ],
    )
    def test_parse(self, raw, base, container):
        ts = TypeSpec.parse(raw)
        assert ts.base_type == base
        assert ts.container == container

    def test_roundtrip_str(self):
        for raw in ("str", "set[str]", "list[float]"):
            assert str(TypeSpec.parse(raw)) == raw

    def test_is_container(self):
        assert TypeSpec.parse("set[str]").is_container
        assert not TypeSpec.parse("str").is_container

    def test_python_type(self):
        assert TypeSpec.parse("float").python_type is float
        assert TypeSpec.parse("set[str]").python_type == set[str]

    @pytest.mark.parametrize("raw", ["", "set[]", "dict[str]", "set{str}", "str[int]"])
    def test_unparseable(self, raw):
        with pytest.raises(ValueError):
            TypeSpec.parse(raw)

    def test_unknown_base_type(self):
        with pytest.raises(ValidationError):
            TypeSpec(base_type="complex")

    def test_unknown_container(self):
        with pytest.raises(ValidationError):
            TypeSpec(base_type="str", container="tuple")

    def test_coerce_scalar(self):
        assert TypeSpec.parse("float").coerce("1.5") == 1.5
        assert TypeSpec.parse("int").coerce("3") == 3

    def test_coerce_date(self):
        assert TypeSpec.parse("date").coerce("2021-01-02") == dt.date(2021, 1, 2)

    def test_coerce_container(self):
        assert TypeSpec.parse("set[str]").coerce(["a", "b", "a"]) == {"a", "b"}
        assert TypeSpec.parse("list[int]").coerce(["1", "2"]) == [1, 2]


class TestParseSpec:
    def test_delimiter_ok(self):
        assert ParseSpec(method="delimiter", pattern=",").pattern == ","

    def test_regex_ok(self):
        spec = ParseSpec(method="regex", pattern=r"(?P<v>\d+)", field="v")
        assert spec.field == "v"

    def test_unknown_method(self):
        with pytest.raises(ValidationError):
            ParseSpec(method="split", pattern=",")

    def test_pattern_required(self):
        with pytest.raises(ValidationError):
            ParseSpec(method="delimiter")

    def test_regex_field_required(self):
        with pytest.raises(ValidationError):
            ParseSpec(method="regex", pattern=r"\d+")


class TestMethodSpec:
    def test_unknown_method(self):
        with pytest.raises(ValidationError):
            MethodSpec(method="fuzzy")

    def test_numeric_method_requires_threshold(self):
        with pytest.raises(ValidationError):
            MethodSpec(method="percent_diff")

    def test_nonnumeric_method_rejects_threshold(self):
        with pytest.raises(ValidationError):
            MethodSpec(method="exact", threshold=0.1)

    def test_numeric_method_with_threshold(self):
        assert MethodSpec(method="range", threshold=5).threshold == 5


class TestColumnSpec:
    def test_single_method(self):
        spec = ColumnSpec(name="c", method="exact")
        assert [m.method for m in spec.methods] == ["exact"]
        assert spec.combinator is None

    def test_any_of(self):
        spec = ColumnSpec(
            name="c",
            type="float",
            any_of=[
                {"method": "percent_diff", "threshold": 0.01},
                {"method": "range", "threshold": 1.0},
            ],
        )
        assert spec.combinator == "any"
        assert len(spec.methods) == 2

    def test_all_of(self):
        spec = ColumnSpec(
            name="c",
            type="float",
            all_of=[
                {"method": "percent_diff", "threshold": 0.01},
                {"method": "range", "threshold": 1.0},
            ],
        )
        assert spec.combinator == "all"
        assert len(spec.methods) == 2

    def test_exactly_one_selector(self):
        with pytest.raises(ValidationError, match="exactly one of"):
            ColumnSpec(name="c")  # none set
        with pytest.raises(ValidationError, match="exactly one of"):
            ColumnSpec(name="c", method="exact", any_of=[{"method": "exact"}])
        with pytest.raises(ValidationError, match="exactly one of"):
            ColumnSpec(
                name="c",
                any_of=[{"method": "exact"}],
                all_of=[{"method": "exact"}],
            )

    def test_column_level_threshold_rejected_for_branches(self):
        with pytest.raises(ValidationError, match="inside each `any_of` branch"):
            ColumnSpec(
                name="c",
                type="float",
                any_of=[{"method": "range", "threshold": 1.0}],
                threshold=0.5,
            )

    def test_delimiter_and_parse_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="not both"):
            ColumnSpec(
                name="c",
                type="set[str]",
                method="exact",
                delimiter=",",
                parse={"method": "delimiter", "pattern": ","},
            )

    def test_container_needs_delimiter(self):
        with pytest.raises(ValidationError, match="needs a `delimiter`"):
            ColumnSpec(name="c", type="set[str]", method="exact")

    def test_delimiter_meaningless_for_scalar(self):
        with pytest.raises(ValidationError, match="meaningless for scalar"):
            ColumnSpec(name="c", type="str", method="exact", delimiter=",")

    def test_effective_parse_from_delimiter(self):
        spec = ColumnSpec(name="c", type="set[str]", method="exact", delimiter="|")
        parse = spec.effective_parse
        assert parse.method == "delimiter" and parse.pattern == "|"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ColumnSpec(name="c", method="exact", bogus=1)


class TestConfig:
    def test_from_dict_injects_names(self):
        cfg = Config.from_dict(
            {"key": "s", "columns": {"c": {"method": "exact", "type": "str"}}}
        )
        assert cfg.key == "s"
        assert cfg.columns["c"].name == "c"

    def test_left_right_key_shared(self):
        cfg = Config.from_dict({"key": "s", "columns": {"c": {"method": "exact"}}})
        assert cfg.left_key == "s" and cfg.right_key == "s"

    def test_left_right_key_pair(self):
        cfg = Config.from_dict(
            {"key1": "a", "key2": "b", "columns": {"c": {"method": "exact"}}}
        )
        assert cfg.left_key == "a" and cfg.right_key == "b"

    def test_key_and_pair_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="not both"):
            Config.from_dict(
                {"key": "s", "key1": "a", "key2": "b", "columns": {}}
            )

    def test_pair_must_be_together(self):
        with pytest.raises(ValidationError, match="set together"):
            Config.from_dict({"key1": "a", "columns": {}})

    def test_unset_key_allowed(self):
        cfg = Config.from_dict({"columns": {"c": {"method": "exact"}}})
        assert cfg.left_key is None

    def test_with_keys_override_single(self):
        cfg = Config.from_dict(
            {"key1": "a", "key2": "b", "columns": {"c": {"method": "exact"}}}
        )
        out = cfg.with_keys(key="s")
        assert out.key == "s" and out.key1 is None and out.key2 is None

    def test_with_keys_override_pair(self):
        cfg = Config.from_dict({"key": "s", "columns": {"c": {"method": "exact"}}})
        out = cfg.with_keys(key1="a", key2="b")
        assert out.key is None and out.left_key == "a" and out.right_key == "b"

    def test_with_keys_noop_returns_self(self):
        cfg = Config.from_dict({"key": "s", "columns": {"c": {"method": "exact"}}})
        assert cfg.with_keys() is cfg

    def test_with_keys_rejects_conflicting_override(self):
        cfg = Config.from_dict({"key": "s", "columns": {"c": {"method": "exact"}}})
        with pytest.raises(ValueError):
            cfg.with_keys(key="s", key1="a", key2="b")

    def test_default_na_values_present(self):
        cfg = Config.from_dict({"key": "s", "columns": {"c": {"method": "exact"}}})
        assert "NA" in cfg.na_values and "" in cfg.na_values
