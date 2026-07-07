"""TheiaValidate v2 — a layered, config-driven table comparison library."""

from theiavalidate.config import ColumnSpec, Config, MethodSpec, ParseSpec, TypeSpec
from theiavalidate.results import ColumnResult, ComparisonResult
from theiavalidate.validator import Validator, compare_tables

__all__ = [
    "Config",
    "ColumnSpec",
    "MethodSpec",
    "ParseSpec",
    "TypeSpec",
    "ColumnResult",
    "ComparisonResult",
    "Validator",
    "compare_tables",
]
