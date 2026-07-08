"""Command-line entry point for theiavalidate.

theiavalidate validate TABLE1 TABLE2 --config validate.yaml
theiavalidate validate TABLE1 TABLE2 --preset theiaprok -- to-come
"""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

import pandas as pd
import rich_click as click

from theiavalidate.config import Config
from theiavalidate.validator import compare_tables

click.rich_click.USE_RICH_MARKUP = True


def _read_table(path: Path) -> pd.DataFrame:
    """Read in TSV"""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _available_presets() -> list[str]:
    try:
        preset_dir = resources.files("theiavalidate") / "presets"
        presets = [p for p in preset_dir.iterdir() if p.name.endswith(".yaml")]
        return sorted(
            preset.name[: -len(".yaml")]
            for preset in presets
        )
    except (FileNotFoundError, NotADirectoryError, ModuleNotFoundError):
        return []


def _load_config(config_path: Path | None, preset: str | None) -> Config:
    if config_path is not None:
        return Config.from_yaml(str(config_path))
    preset_candidate = resources.files("theiavalidate") / "presets" / f"{preset}.yaml"
    if not preset_candidate.is_file():
        available = _available_presets()
        raise click.UsageError(
            f"unknown preset {preset!r}; available: {', '.join(available) or 'none bundled yet'}"
        )
    return Config.from_yaml(str(preset_candidate))


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
    "--preset", help="Name of a bundled workflow preset (alternative to --config)."
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
    help="Also write a PDF report (needs the `report` extra + wkhtmltopdf).",
)
@click.option(
    "--exit-zero",
    is_flag=True,
    help="Always exit 0, even when differences are found.",
)
def validate(
    table1: Path,
    table2: Path,
    config_path: Path | None,
    preset: str | None,
    outdir: Path,
    prefix: str,
    html: bool,
    pdf: bool,
    exit_zero: bool,
) -> None:
    """Compare TABLE1 and TABLE2 using a config (or preset)."""
    if bool(config_path) == bool(preset):
        raise click.UsageError("provide exactly one of --config or --preset")

    try:
        config = _load_config(config_path, preset)
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
        click.secho("PASSED", fg="green", bold=True)
    else:
        click.secho("DIFFERENCES FOUND", fg="red", bold=True)
        if not exit_zero:
            sys.exit(1)


if __name__ == "__main__":
    main()
