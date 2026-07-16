"""Command-line entry point for theiavalidate.

theiavalidate validate TABLE1 TABLE2 --config validate.yaml
theiavalidate validate TABLE1 TABLE2 --preset theiaprok_pe
theiavalidate validate TABLE1 TABLE2 --preset theiaprok_pe --preset-dir ./presets
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import rich_click as click

from theiavalidate.config import Config
from theiavalidate.presets import resolve_preset
from theiavalidate.validator import compare_tables

click.rich_click.TEXT_MARKUP = "rich"


def _read_table(path: Path) -> pd.DataFrame:
    """Read in TSV"""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _load_config(
    config_path: Path | None,
    preset: str | None,
    preset_dir: Path | None,
    preset_ref: str,
    refresh: bool,
) -> Config:
    if config_path is not None:
        return Config.from_yaml(str(config_path))

    yaml_path = resolve_preset(
        preset, preset_dir=preset_dir, ref=preset_ref, refresh=refresh
    )
    return Config.from_yaml(str(yaml_path))


@click.group()
@click.version_option(package_name="theiavalidate")
def main() -> None:
    """TheiaValidate ~~ config-driven comparison of tabular pipeline outputs."""


@main.command()
@click.argument("table1", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("table2", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a YAML config.",
)
@click.option(
    "--preset",
    help=(
        "Named workflow preset, e.g. theiaprok_pe (alternative to --config). "
        "Resolved from --preset-dir if given, otherwise downloaded from the "
        "theiagen public_health_bioinformatics repo and cached."
    ),
)
@click.option(
    "--preset-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Local directory of preset YAMLs to resolve --preset against (skips download).",
)
@click.option(
    "--preset-ref",
    default="main",
    show_default=True,
    help="Git ref (branch/tag/commit) to download presets from.",
)
@click.option(
    "--refresh",
    is_flag=True,
    help="Re-download a cached preset instead of reusing the cached copy.",
)
@click.option(
    "--key",
    help="Join key column shared by both tables (overrides the config/preset).",
)
@click.option(
    "--key1",
    help="Join key column in TABLE1 (overrides the config/preset; use with --key2).",
)
@click.option(
    "--key2",
    help="Join key column in TABLE2 (overrides the config/preset; use with --key1).",
)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Directory for output files.",
)
@click.option(
    "--prefix",
    default="theiavalidate",
    show_default=True,
    help="Output filename prefix.",
)
@click.option(
    "--html/--no-html", default=True, show_default=True, help="Write an HTML report."
)
@click.option(
    "--pdf",
    is_flag=True,
    help="Also write a PDF report (needs the wkhtmltopdf system binary).",
)
def validate(
    table1: Path,
    table2: Path,
    config_path: Path | None,
    preset: str | None,
    preset_dir: Path | None,
    preset_ref: str,
    refresh: bool,
    key: str | None,
    key1: str | None,
    key2: str | None,
    outdir: Path,
    prefix: str,
    html: bool,
    pdf: bool,
) -> None:
    """Compare TABLE1 and TABLE2 using a config (or preset)."""
    if bool(config_path) == bool(preset):
        raise click.UsageError("provide exactly one of --config or --preset")
    if config_path is not None and (preset_dir is not None or refresh):
        raise click.UsageError("--preset-dir/--refresh only apply with --preset")

    try:
        config = _load_config(config_path, preset, preset_dir, preset_ref, refresh)
        config = config.with_keys(key=key, key1=key1, key2=key2)
        left = _read_table(table1)
        right = _read_table(table2)
        result = compare_tables(
            left, right, config, left_name=table1.name, right_name=table2.name
        )
        result.write(str(outdir), prefix=prefix, tsv=True, html=html, pdf=pdf)
    except (OSError, ValueError) as err:
        raise click.ClickException(str(err))

    click.echo(result.summary_df().to_string())
    click.echo(f"\nOutput written to {outdir}/")
    if result.passed:
        click.secho("PASSED, LGTM", fg="green", bold=True)
    else:
        click.secho("DIFFERENCES FOUND", fg="red", bold=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
