"""Declarative config for a TheiaValidate comparison.

The YAML describes the keys and columns to compare, along with the comparison method and type.
The YAML models the columns to compare and their comparison methods, along with any custom type adapters.

The config returns a `ComparisonConfig` object that can be used to compare two datasets. Holding the rules for comparison.

ConfigSpec: Holds the overall configuration for a comparison, including the columns to compare and their comparison methods.

ColumnSpec: Holds the configuration for a single column, including its name, type, and comparison method.

TypeSpec: Holds the expected Python type of a column, including its base type and any container type(if present).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from functools import lru_cache
from typing import Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    TypeAdapter,
    field_validator,
    model_validator,
)

# Comparison methods we want to support. This will expand over time.
KNOWN_METHODS = frozenset(
    {"exact", "percent_diff", "range", "ignore", "file_exact"}
)

# Methods that require a numeric threshold
NUMERIC_METHODS = frozenset({"percent_diff", "range"})


_DATA_TYPES = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "date": date,
    "datetime": datetime,
}
_CONTAINERS = {"set": set, "list": list}


@lru_cache(maxsize=None)
def _adapter_for(python_type) -> TypeAdapter:
    """One cached TypeAdapter per resolved type.

    Example:
        first call:  _adapter_for(set[float])  → not in cache → runs TypeAdapter(set[float]) → stores it → returns it
        later call:  _adapter_for(set[float])  → already in cache → skips the body → returns the stored adapter
    """
    return TypeAdapter(python_type)


# Values treated as null before comparison. A column may extend/override this. Expanded
# from the previous codebase.
DEFAULT_NA_VALUES = ["", "NA", "N/A", "n/a", "NaN", "nan", "None", "null", "NULL"]


def _check_key_pair(
    key: Optional[str], key1: Optional[str], key2: Optional[str]
) -> None:
    """Enforce the key mutual-exclusion rules (shared by config + CLI override).

    A single shared `key` and the per-table `key1`/`key2` pair are mutually
    exclusive; the pair must be given together. An entirely unset key is allowed.
    """
    if key is not None and (key1 is not None or key2 is not None):
        raise ValueError("set either `key` or `key1`+`key2`, not both")
    if (key1 is None) != (key2 is None):
        raise ValueError("`key1` and `key2` must be set together")


# Regex for parsing type strings like `set[str]`, `list[float]`, `str`. The
# optional container wraps the base type; the conditional requires the closing
# `]` only when a container opened one, so `str` and `set[str]` share the single
# `base_type` group (Python's re forbids reusing a group name across branches).
_TYPE_RE = re.compile(
    r"^(?:(?P<container>set|list)\[)?(?P<base_type>\w+)(?(container)\])$"
)


class TypeSpec(BaseModel):
    """The expected Python type of a column. Can be a scalar primitive or a container of scalars.

    Parsed from strings like `str`, `float`, `set[str]`, `list[float]`. Exposes
    the base (scalar) type separately so parsing can split-then-coerce each element.
    """

    model_config = ConfigDict(frozen=True)

    base_type: str  # one of _DATA_TYPES
    container: Optional[str] = None  # None (scalar), "set", or "list"

    @classmethod
    def parse(cls, raw: str) -> "TypeSpec":
        match = _TYPE_RE.match(raw.strip())
        if not match:
            raise ValueError(
                f"unparseable type {raw!r} (expected e.g. str, float, set[str], list[float])"
            )
        return cls(
            base_type=match.group("base_type"), container=match.group("container")
        )

    @field_validator("base_type")
    @classmethod
    def _known_base_type(cls, v: str) -> str:
        if v not in _DATA_TYPES:
            raise ValueError(
                f"unknown base type {v!r}; expected one of {sorted(_DATA_TYPES)}"
            )
        return v

    @field_validator("container")
    @classmethod
    def _known_container(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _CONTAINERS:
            raise ValueError(
                f"unknown container {v!r}; expected one of {sorted(_CONTAINERS)}"
            )
        return v

    @property
    def is_container(self) -> bool:
        return self.container is not None

    @property
    def python_type(self):
        """Resolve to the real Python type: str, float, date, set[str], ..."""
        base_type = _DATA_TYPES[self.base_type]
        return _CONTAINERS[self.container][base_type] if self.container else base_type

    def coerce(self, value: object) -> object:
        """Coerce via pydantic. Pass a scalar for a scalar type, or the already-
        split list of elements for a container type; pydantic coerces each element
        and assembles the set/list."""
        return _adapter_for(self.python_type).validate_python(value)

    def __str__(self) -> str:
        return (
            f"{self.container}[{self.base_type}]" if self.container else self.base_type
        )


class ParseSpec(BaseModel):
    """How to turn a raw cell into a comparable value before comparison.

    `delimiter` -> split into a collection; `regex` -> extract a named group.
    """

    # Con't afford to provide extra fields, as it might allow unexpected fields to be passed
    model_config = ConfigDict(extra="forbid")

    method: str  # "delimiter" | "regex"
    pattern: Optional[str] = None  # the delimiter string, or the regex
    field: Optional[str] = None  # named regex group to extract

    @field_validator("method")
    @classmethod
    def _known_parse_method(cls, v: str) -> str:
        if v not in {"delimiter", "regex"}:
            raise ValueError(f"unknown parse method {v!r}; expected delimiter or regex")
        return v

    @model_validator(mode="after")
    def _requirements(self) -> "ParseSpec":
        if not self.pattern:
            raise ValueError(f"parse method {self.method!r} requires a `pattern`")
        if self.method == "regex" and not self.field:
            raise ValueError(
                "parse method 'regex' requires a `field` (the named group to extract)"
            )
        return self

    @property
    def yields_collection(self) -> bool:
        # A delimiter split always produces a collection; regex produces a scalar (for now)
        return self.method == "delimiter"


class MethodSpec(BaseModel):
    """A single comparison: a method plus its threshold (if numeric).

    Used both for a column's top-level comparison and for each branch of
    `any_of`/`all_of`.
    """

    model_config = ConfigDict(extra="forbid")

    method: str
    threshold: Optional[float] = None

    @field_validator("method")
    @classmethod
    def _known_method(cls, v: str) -> str:
        if v not in KNOWN_METHODS:
            raise ValueError(
                f"unknown method {v!r}; expected one of {sorted(KNOWN_METHODS)}"
            )
        return v

    @model_validator(mode="after")
    def _threshold_rules(self) -> "MethodSpec":
        if self.method in NUMERIC_METHODS and self.threshold is None:
            raise ValueError(f"method {self.method!r} requires a numeric `threshold`")
        if self.method not in NUMERIC_METHODS and self.threshold is not None:
            raise ValueError(f"method {self.method!r} does not take a `threshold`")
        return self


class ColumnSpec(BaseModel):
    """One column's comparison rule."""

    model_config = ConfigDict(extra="forbid")

    name: str  # column name placeholder
    type: TypeSpec = TypeSpec(base_type="str")
    method: Optional[str] = None
    threshold: Optional[float] = None
    any_of: Optional[list[MethodSpec]] = None
    all_of: Optional[list[MethodSpec]] = None
    delimiter: Optional[str] = None
    parse: Optional[ParseSpec] = None
    mappings: list[str] = []  # alternate source column names in either table
    na_values: Optional[list[str]] = None  # per-column extension of the global set

    _methods: list[MethodSpec] = PrivateAttr(default_factory=list)
    _combinator: Optional[str] = PrivateAttr(default=None)  # "any", "all", or None

    @field_validator("type", mode="before")
    @classmethod
    def _parse_type(cls, v: object) -> object:
        return TypeSpec.parse(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _validate(self) -> "ColumnSpec":
        # Exactly one of `method`, `any_of`, or `all_of` selects the comparison.
        selectors = [n for n in ("method", "any_of", "all_of") if getattr(self, n)]
        if len(selectors) != 1:
            raise ValueError(
                f"column {self.name!r}: set exactly one of `method`, `any_of`, or `all_of`"
            )

        if self.method:
            self._methods = [MethodSpec(method=self.method, threshold=self.threshold)]
        else:
            branches = self.any_of or self.all_of
            if self.threshold is not None:
                kind = "any_of" if self.any_of else "all_of"
                raise ValueError(
                    f"column {self.name!r}: put `threshold` inside each `{kind}` branch, not at column level"
                )
            self._methods = list(branches or [])
            self._combinator = "any" if self.any_of else "all"

        # delimiter and parse are mutually exclusive
        if self.delimiter is not None and self.parse is not None:
            raise ValueError(
                f"column {self.name!r}: set either `delimiter` or `parse`, not both"
            )

        # Couple the collection-builder to the container type.
        builds_collection = self.delimiter is not None or (
            self.parse is not None and self.parse.yields_collection
        )
        if self.type.is_container and not builds_collection:
            raise ValueError(
                f"column {self.name!r}: type {self.type} needs a `delimiter` (or a collection-yielding `parse`)"
            )
        if not self.type.is_container and self.delimiter is not None:
            raise ValueError(
                f"column {self.name!r}: `delimiter` is meaningless for scalar type {self.type}"
            )

        builds_collection = self.delimiter is not None or (
            self.parse is not None and self.parse.yields_collection
        )
        if self.type.is_container and not builds_collection:
            raise ValueError(
                f"column {self.name!r}: type {self.type} needs a `delimiter` (or a collection-yielding `parse`)"
            )
        if not self.type.is_container and self.delimiter is not None:
            raise ValueError(
                f"column {self.name!r}: `delimiter` is meaningless for scalar type {self.type}"
            )
        return self

    @property
    def methods(self) -> list[MethodSpec]:
        """The comparison(s) to run, always a list, whether `method`, `any_of`, or `all_of`."""
        return self._methods

    @property
    def combinator(self) -> Optional[str]:
        """How to fold multiple branches: "any" (OR), "all" (AND), or None for a single method."""
        return self._combinator

    @property
    def effective_parse(self) -> Optional[ParseSpec]:
        """Normalize `delimiter` into a ParseSpec so parsing has one path."""
        if self.parse is not None:
            return self.parse
        if self.delimiter is not None:
            return ParseSpec(method="delimiter", pattern=self.delimiter)
        return None


class Config(BaseModel):
    """A full comparison config: the join key plus per-column rules.

    The join key can be given two ways:
      - `key` — a single column name shared by both tables (the common case).
      - `key1` + `key2` — a per-table key, for when the two tables name their
    """

    model_config = ConfigDict(extra="forbid")

    key: Optional[str] = None
    key1: Optional[str] = None
    key2: Optional[str] = None
    na_values: list[str] = DEFAULT_NA_VALUES
    columns: dict[str, ColumnSpec]

    @model_validator(mode="after")
    def _key_rules(self) -> "Config":
        # Note: an entirely unset key is allowed here, presets won't have one
        _check_key_pair(self.key, self.key1, self.key2)
        return self

    @property
    def left_key(self) -> Optional[str]:
        """Key column name in the left (first) table, if resolved."""
        return self.key1 if self.key1 is not None else self.key

    @property
    def right_key(self) -> Optional[str]:
        """Key column name in the right (second) table, if resolved."""
        return self.key2 if self.key2 is not None else self.key

    def with_keys(
        self,
        *,
        key: Optional[str] = None,
        key1: Optional[str] = None,
        key2: Optional[str] = None,
    ) -> "Config":
        """Return a copy with the join key overridden (e.g. from the CLI).

        Any override fully replaces the config's key. A single `key` and the
        `key1`/`key2` pair remain mutually exclusive. No override returns self.
        """
        if key is None and key1 is None and key2 is None:
            return self
        _check_key_pair(key, key1, key2)
        if key is not None:
            return self.model_copy(update={"key": key, "key1": None, "key2": None})
        return self.model_copy(update={"key": None, "key1": key1, "key2": key2})

    @model_validator(mode="before")
    @classmethod
    def _inject_column_names(cls, data: object) -> object:
        # Give each ColumnSpec its name from the mapping key.
        if isinstance(data, dict) and isinstance(data.get("columns"), dict):
            for key, spec in data["columns"].items():
                if isinstance(spec, dict):
                    spec.setdefault("name", key)
        return data

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return cls.model_validate(data)
