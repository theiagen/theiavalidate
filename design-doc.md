# TheiaValidate v2 Design Proposal

## Purpose

This document describes the v2 design for TheiaValidate. The current tool compares two tab-delimited tables and reports their differences, but its logic is rigid. Comparisons are matched by row position, comparison type is inferred from a cell's Python dtype, and structured fields (BUSCO, AMRFinder) are parsed with some regex hacks. v2 reorganizes the tool into a layered, importable library driven by a single declarative config, so comparisons generalize cleanly and the library can be executed standalone or ported to`bioblueprint`/`bioforklift`.

## Scope

Applies to the TheiaValidate rewrite only.  The comparison logic, config format, Python API, output/reporting, and an optional LLM summary layer (secondary). We want to make the underlying validation module is more flexible while also considering how to better support our partners validation. 

## Goals

1. Compare tables of **different shapes / column counts** without silent misalignment.
2. Strict python type coercion
3. **Comma-separated cell** comparison as a first-class option (set or sorted-list).
4. **Generalized parsing** of structured but uninformative fields (BUSCO, AMR) via delimiter/regex/named parsers and not per-field special cases.
5. A **simple Python API** with typed inputs, importable into `bioblueprint`/`bioforklift`.
6. **Percent-difference output** surfaced alongside pass/fail, for cross-result context.
7. One **YAML config** (field name, type, threshold, parsing) consumed by pure comparison functions wrapped in a simple class leveraging composition.

## Architecture

The proposed architecture is a one-directional pipeline of small modules. The comparison path is fully deterministic; the LLM layer is an optional consumer of its output and will probably come as an independent add on. 

| Module | Responsibility |
| --- | --- |
| `config.py` | Load + validate the YAML into pydantic models (`Config`, `ColumnSpec`, `ParseSpec`). |
| `parsing.py` | Parser registry: `identity` / `delimiter` / `regex` / `named` (e.g. `busco`, `amrfinder`). |
| `comparators.py` | Pure comparison functions + registry: `exact`, `percent_diff`, `range`, `ignore`, `file_*`. |
| `alignment.py` | Join the two tables on the key column; report row/column exclusives. |
| `results.py` | `ColumnResult` / `ComparisonResult` containers + `summary_df()`, `to_dict()`, `write()`. |
| `validator.py` | Orchestrator (`Validator`) + `compare_tables()` one-call shortcut. |
| `reporting.py` | Render HTML or PDF from a `ComparisonResult`. |
| `agent.py` | **Optional** summarizer + faithfulness check. Imports `anthropic`; installed as an extra. |
| `cli.py` | argparse → `Config` → `Validator` → `result.write()`. |

### Data flow

```
tables ─► align (key-join) ─► parse ─► compare ─► ComparisonResult ─► HTML / TSV / dict   (deterministic)
```

## 1. Configuration (single YAML)

One file declares the key column and, per column, the comparison type, threshold, parsing, and any aliases. It replaces the three current inputs (`columns_to_compare`, `validation_criteria`, `column_translation`).

```yaml

key: sample_id
na_values: [None, "NA", ""]          # note: unquoted None -> string "None", not null

columns:
  predicted_taxon:
    method: exact
    type: str
    mappings: [gambit_predicted_taxon]

  amrfinderplus_amr_core_genes:
    method: exact
    delimiter: ","                    # comma-separated cell
    type: set[str]
    mappings: [amrfinderplus_amr_genes]

  assembly_length:
    method: percent_diff
    threshold: 0.01

  busco_results:                      # the "uninformative field" case
    type: set[float]
    parse:
      method: regex
      pattern: 'C:(?P<complete>[\d.]+)%'
      field: complete
    any_of:
      - method: range
        threshold: 2
```

Note: use `method` instead of `type` , `type` needs to be reserved for python coercion

Note: have general set of null, but also per column null guards 

Use the block method for annotating 

Each column will have the method, this is the the actual comparison that will be done. Each column will have a type, a type will be the expected python type
Columns that get converted to lists or sets will need the delimiter to be parsed by. 
Columns that have a list or set, need to have some sort of subtype config that correctly converts / coerces those values into the pythonic type.  This will be done in the format of list[type]/set[type].

One thing we could do as well, is have presets per workflow that could be called as part of the default suite. So we could run `theivalidate validate theiaprok` or something like that. Otherwise we rip `theiavalidate validate --config path/to.custom.yaml`

## 2. Alignment

Both tables are **outer-joined on the key column**. Rows or columns present in only one table are reported as exclusives, not silently filled and compared.
 The current codebase compares row *N* of table1 to row *N* of table2; two tables with the same samples in a different order produce meaningless diffs. Joining on the key makes differing row counts and orders a normal case and removes the latent positional bug. Or put more digestibly, order shouldn’t matter, just match on key (sample_id)

## 3. Parsing

A `parse:` block converts a cell to a comparable value before the  comparison is done:

- **`delimiter`** — split into a list (AMR genes → list/set).
- **`regex`** — extract a named group as the comparable scalar (BUSCO completeness `C:96.5%` → `96.5`).

Extracting the comparable value is separated from comparing it, so a new structured field is a one-function parser contribution rather than another branch in the comparison logic.

## 4. Comparators (pure functions)

**`method` and `type` are distinct.** `method` is the comparison to run (`exact`, `percent_diff`, `range`, `ignore`, `file_*`); `type` is the expected Python type each cell is coerced to first. The pipeline is always **coerce (`type`) → compare (`method`)**. A registry maps the config `method` string to the function.

```python
def exact(left, right, *, spec) -> ColumnResult: ...
def percent_diff(left, right, *, spec) -> ColumnResult: ...
```

Because coercion happens first, there is **no separate `set` method** — a column with `type: set[str]` and `method: exact` is set equality (Python `set == set` ignores order). The comparator only needs to be type-aware for **diff rendering**: `exact` on two differing sets reports the symmetric difference (which items were added/removed), not `"a,b,c" vs "a,c,b"`. (`set` drops duplicates; use `type: list[str]` + a sort flag if duplicates must be preserved.)

Numeric comparators always compute the **percent difference per row** into `ColumnResult.measures`, so the summary can report max/mean % diff per column even when the method is `exact`.

## 5. API

DataFrames from bioforklift or table as tsv paths as input, a `ComparisonResult` object as output. 

```python
from theiavalidate import compare_tables, Config

cfg = Config.from_yaml("validate.yaml")
result = compare_tables(dev_df, main_df, cfg)   # the dev-vs-main
assert result.passed                            # deterministic gate
result.summary_df()                             # per-column counts + % diff
result.write("out/", html=True, pdf=False)      # optional outputs
```

`Validator(cfg).compare(t1, t2)` is the class form; `compare_tables` is the shortcut. `ComparisonResult` composes the per-column `ColumnResult`s and the alignment exclusives.

## 6. Optional LLM summary

A summarizer turns a `ComparisonResult` into a short plain-language readout.  It reads the computed summary  and is never part of the comparison.

```python
from theiavalidate.agent import summarize        
summary = summarize(result, model="claude-opus-4-8")   # or AnthropicVertex client (GCP ADC)
```

Mismatches are flagged in the report. I think we can expand this idea a bit further, but I need to play around with this a bit more. I think the determinisitic path will shine insight onto the supplementary llm part. 

## Development Path

1. Build the deterministic core logic (`config` → `parsing` → `comparators` → `alignment` → `results` → `validator`) with unit tests per pure function. This would expand the testing suite and enforce CI/CD around github to adhere to passing tests. 
2. `cli.py` accepts a custom input `yaml` or a preset workflow and writes the same output filenames, so the entry point could be one path or preset. 
3. Ship a `legacy_to_yaml` converter that emits the new config from the three old TSVs (if we think this is needed)
4. Deprecate the `gs://`autodetect and dtype-inferred-criteria paths.
5. Add the optional `agent.py` layer once the logic is stable. Probably as another PR.
