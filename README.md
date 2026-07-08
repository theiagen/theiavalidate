# theiavalidate

Theiavalidate v2 is meant to be a config driven comparison of two tabular pipeline outputs with the configuration specified in a YAML file. The configuration file defines the columns to compare and the comparison methods to use. The idea is to make it easy to compare two tabular pipeline outputs without writing custom comparison scripts. While we mainly use this for validation, there are areas outside of validation that require comparsison of pipeline outputs.

> v2 rewrite. Requires Python ≥ 3.12.

## Install

```bash
uv sync  
# or
pip install .
```

PDF output (`--pdf`) also needs the `wkhtmltopdf`.

## Run

Use a bundled **preset** or a custom `--config` YAML. Tables are joined on a key
column; because Terra names it `entity:<table>_id`, you can pass the
key(s) at run time:

```bash
theiavalidate validate TABLE1.tsv TABLE2.tsv --preset theiaprok_pe \
  --key1 entity:<table1>_id --key2 entity:<table2>_id
```

- `--key1/--key2` — per-table key when the columns differ (`key1` = TABLE1).
- `--key` — single key column shared by both tables.
- CLI keys override any key set in the config.

Custom config and other options:

```bash
theiavalidate validate a.tsv b.tsv --config my.yaml \
  --outdir out --prefix run1 --pdf -
```

`--outdir` (default `.`), `--prefix`, `--html/--no-html` (default on), `--pdf`,

### Python API

```python
from theiavalidate import compare_tables, Config

cfg = Config.from_yaml("my.yaml")
result = compare_tables(dev_df, main_df, cfg)   # pandas DataFrames in
assert result.passed
result.summary_df()
result.write("out/", prefix="run1", html=True)
```

## Data flow

```
TABLE1 ┐
       ├─ align ──► parse ──► coerce ──► compare ──► ComparisonResult ──► TSV / HTML / PDF
TABLE2 ┘  (key-join)  (split/  (to type)  (method)      (pass/fail
                       regex)                            + measure)
```

- **align** — outer-join on the key; shared rows are compared, exclusive rows/columns reported.
- **parse** — split a cell on a `delimiter`, or extract a `regex` group.
- **coerce** — cast to the column's `type` (`str`, `float`, `set[str]`, …).
- **compare** — run the column's `method`; a row passes if it matches / is within threshold.

## Config

```yaml
key: sample_id            # or key1/key2 for differing per-table columns
columns:
  gambit_predicted_taxon:
    method: exact
  assembly_length:
    type: float
    any_of:               # passes if EITHER branch passes
      - method: percent_diff 
        threshold: 0.01
      - method: range
        threshold: 500
  amrfinderplus_amr_core_genes:
    method: exact
    type: set[str]         # order-independent set equality
    delimiter: ","
```

- `method`: how to compare a column
- `type`: python value to coerce into
- `mappings`: `na_values`, `delimiter`/`parse`
- `any_of`: list of `method`/`type` pairs to try in order
- `delimiter`: custom delimiter for `method: exact` with `type: set[str]`

## Methods

| method | use | notes |
|---|---|---|
| `exact` | equality | any type; `set[str]` → set equality |
| `percent_diff` | numeric within % | `threshold` as fraction (0.01 = 1%); scalar numeric only |
| `range` | numeric/date within absolute delta | `threshold` in value units (days for dates); scalar only |
| `ignore` | skip | always passes, in case we want to keep a column in the config, but for the comparison we want to skip it |
| `file_exact` | file content equality | md5 of local path or `gs://`/`s3://` URI |

Where both cells are null, they match. Where one is null we get a mismatch. Combine methods with `any_of`.

## Output

Under `--outdir`:

- `<prefix>_summary.tsv` — per-column method, rows compared, differences
- `<prefix>_differences.tsv` — differing rows with both values
- `<prefix>_report.html` / `.pdf` — renderd report

## Presets

Bundled in `src/theiavalidate/presets/`, named `<workflow>_<readtype>`:

`theiaprok_{pe,se,ont,fasta}`, `theiacov_{pe,se,ont,fasta,clearlabs}`,
`theiaeuk_{pe,ont}`, `theiameta_pe`, `theiaviral_{pe,ont,panel}`.

## Test inputs
Find phb test inputs in tests/phb/*, and use test_run.sh as the driver


## Developer Notes

For this first round I focused on taking the methods that existed in the v1 version and moving them over to the new format. However, there are certain things I haven't made decisions on yet. For example container types like set/list can only be split and compared as `exact`. The reason for this is because we can't trust the data between two different workflow versions will capture the data the same way. So how do we best handle container types like set/list on a more granular level? Or is checking for equality enough? 

For now the expected input is a TSV file, but with bioforklift we could pull the table down directly. I think that's an easy addition. Functionality is more important right now. 

But generally, what other methods do we want to add? Is there nuance that isn't being capture here. I mostly transitioned known methods from v1 to the new format, but there are certainly more to add.


Please review carefully, my eyes burn.
