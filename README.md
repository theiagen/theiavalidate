# theiavalidate

Note: this repository is undergoing active development. Check back for updates.

## Docker

We recommend using our Docker image to run this tool as all dependencies are installed for your convenience.

```bash
docker pull us-docker.pkg.dev/general-theiagen/theiagen/theiavalidate:1.1.3
```

## Dependencies

To use this tool on the command line, please ensure all dependencies are installed. Feel free to use the pyproject.toml file within this repository as demonstrated by executing the following command within the repo directory:

```
pip3 install .
```

To generate the PDF report, please ensure the following is installed as well:

```
apt-get install wkhtmltopdf
```

## Usage

```text
usage: python3 theiavalidate.py table1 table2 [options]

This tool compares two tab-delimited files and outputs a report of the differences between the two files.

positional arguments:
  table1  the first table to compare
  table2  the second table to compare

optional arguments:
  -h, --help
          show this help message and exit
  -v, --version
          show program's version number and exit
  -c, --columns_to_compare
          a comma-separated list of columns to compare
          required for a successful run
  -m, --validation_criteria
          a tab-delimited file containing the validation criteria to check
  -l, --column_translation
          a tab-delimited file that links column names between the two tables
  -o, --output_prefix
          the output file name prefix
          do not include any spaces
  -n, --na_values
          a comma-delimited string of values that should be considered NA
          default values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', '', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', 'None']
  --verbose
          increase stdout verbosity
  --debug
          increase stdout verbosity to debug; overwrites --verbose
```

### Inputs Explained

See also the [`examples`](https://github.com/theiagen/theiavalidate/tree/main/examples) folder for example inputs.

#### Required: `table1` and `table2`

These are the two TSV files that will be examined. The order of the tables does not matter.

**CAUTION:** each table requires exactly the same number of samples and matching sample names (or values in the first column). If the tables do not have the same samples, the script will fail. There can be no additional samples in either table as well.

#### Required: `columns_to_compare`

The `columns_to_compare` variable determines what columns will be examined. This is a comma separated list, such as: `"assembly_length,est_coverage,gambit_predicted_taxon"`. The order of the columns does not matter. All other columns not listed will be ignored.

#### Optional: `validation_criteria`

An example validation_criteria.tsv file is shown below. The first column is the column name in the two tables. The second column is the validation criteria to use for that column. The third column can contain any non-standard delimiters that are found for that particular column (e.g., ":" or "/"). This file **expects a header** and is **tab-delimited**. The text in the header does not matter.

**CAUTION:** Any column names in this file must also be in `columns_to_compare` for additional validation criteria to be performed.

```text
column_name     validation_criteria     delimiter
column1         EXACT                   
column2         SET                     ";"
column3         0.01                    
```

Currently implemented validation criteria include:

| validation_criteria | explanation |
| --- | --- |
| EXACT | The values in the two columns must be exactly the same; in this case `[foo,bar] != [bar,foo]`. When applied to columns referencing files, file contents will be compared to check if they are identical.|
| SET | The values in the two columns must be the same set of values; in this case `[foo,bar] == [bar,foo]`. When applied to columns referencing files, the lines within the files will be sorted alphabetically before comparing.|
| \<INT\> | The values in the two columns must be within `<INT>` of each other; e.g., 50 -> 50 units apart allowed. |
| \<FLOAT\> | The values in the two columns must be within `<FLOAT>*100` of each other; e.g., 0.3 -> 30% difference allowed. |
| IGNORE | The values in the two columns are assumed to match; in this case `foo == bar`. |
| CRITERIA1,CRITERIA2,... | The values in the two columns will be compared with CRITERIA1 (as indicated above) and _then_ with CRITERIA2, etc.; values pass if at least one criteria are met; the separate criteria *must* be comma-delimited. |

#### Optional: `column_translation`

An example column_translation.tsv file is shown below. The first column is the column name in one table, and the second column is the corresponding column name in the other table. _All columns with the name in the first column will be renamed to match the corresponding column name in the second column_. This file **expects a header** and is tab-delimited. The text within the header does not matter and is not used.

```
old_name               new_name
column_name1_table1    column_name1_table2
column_name2_table1    column_name2_table2
original_column_name   new_column_name
```

For example, if `table1` has a column named `column_name1_table1`, it will be renamed to `column_name1_table2` in all outputs and comparisons.

#### Optional: `output_prefix`

The output prefix variable is a string that will prefix all output file. Do not include any whitespace. The default is `theiavalidate`.

#### Optional: `na_values`

The `na_values` variable is a list of values that should be considered NA by Pandas. The default list is _different_ than the default na_values list used by Pandas. This is because some outputs are legitimately `"NA"` and should not be considered missing data by Pandas. All and _only_ the values in this list will be replaced with `pandas.na` or `numpy.nan` in the output files and comparisons.

#### Optional: `verbose` and `debug`

These two outputs increase the verbosity of the logging system to `INFO` and `DEBUG`, respectively. `DEBUG` produces far more output than `INFO` and may be excessive for non-debugging purposes. If both `--debug` and `--verbose` are present, `--debug` takes precendence. If no verbosity is specified, the logging level is set to `ERROR`.

### Outputs Explained

See also the [`examples`](https://github.com/theiagen/theiavalidate/tree/main/examples) folder for example outputs.

Or, you can copy and paste following command in the Docker image to generate the example outputs.

```bash
theiavalidate.py \
  theiavalidate/examples/example-table1.tsv \
  theiavalidate/examples/example-table2.tsv \
  -c "assembly_length,gambit_predicted_taxon,amrfinderplus_amr_core_genes,extra_column" \
  -l theiavalidate/examples/example-column_translation.tsv \
  -m theiavalidate/examples/example-validation_criteria.tsv \
  -o example-output
```

#### `filtered_<table1_name>` and `filtered_<table2_name>`

These files are the original input files with _only_ the columns specified in `columns_to_compare` and all columns being renamed to what is specified in the `column_translation.tsv` file. These files are provided to allow the user to see what columns are being compared and to allow the user to manually inspect the original data.

#### `<output_prefix>_exact_differences.tsv`

This file is a tab-delimited file containing all rows and columns specified in `columns_to_compare`. The only values in this file are the values that are not exactly the same between the two tables. 

#### `<output_prefix>_validation_criteria_differences.tsv`

**NOTE:** This file is only provided if a `validation_criteria.tsv` file is provided. This file is a tab-delimited file containing all rows and columns specified in `columns_to_compare`. The only values in this file are the values that do not meet the validation criteria specified in the `validation_criteria.tsv` file.

Columns with all values passing validation criteria are excluded. For set comparisons, _only the differences are displayed_; i.e., the items for the column in table1 that were not found in table2 will be found under the table1 column and vice versa. Please note that either of these behaviors may not apply for rows with multiple criteria (`SET,0.05`).

#### `<output_prefix>_summary.html` and `<output_prefix>_summary.pdf`

This file (available as an HTML and PDF) is a summary of the differences between the two tables. It contains the following information:

- the date `theiavalidate.py` was run
- as rows, the columns specified in `columns_to_compare`
- as columns:
  - the number of rows in `table1` that have values
  - the number of rows in `table2` that have values
  - the number of differences (exact match)
  - the corresponding validation criteria (if provided)
  - the number of samples failing the validation criteria

If a `validation_criteria.tsv` file was provided, a definition of the (currently implemented) validation criteria are provided at the bottom of the table

#### `<sample>_<column>_diff.txt`
Shows the differing lines within mismatching files for a given sample and column. Each pair of mismatching files generates a separate file.
