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
        return sorted(preset.name[: -len(".yaml")] for preset in presets)
    except (FileNotFoundError, NotADirectoryError, ModuleNotFoundError):
        return []


def _criteria_source(config_path: Path | None, preset: str | None) -> tuple[Path, str]:
    """Resolve the criteria YAML path and a workflow label (config stem / preset name).

    Shared by config loading and the optional agent layer, which reads the same YAML.
    """
    if config_path is not None:
        return config_path, config_path.stem
    preset_candidate = resources.files("theiavalidate") / "presets" / f"{preset}.yaml"
    if not preset_candidate.is_file():
        available = _available_presets()
        raise click.UsageError(
            f"unknown preset {preset!r}; available: {', '.join(available) or 'none bundled yet'}"
        )
    return Path(str(preset_candidate)), preset


def _load_config(config_path: Path | None, preset: str | None) -> Config:
    criteria_path, _ = _criteria_source(config_path, preset)
    return Config.from_yaml(str(criteria_path))


_DEFAULT_QUESTION = (
    "Summarize the differences between the two tables. For each column that differs, "
    "say whether it looks like noise within threshold or a real regression, and flag "
    "any rows or columns exclusive to one table."
)


def _run_agent(result, criteria_path: Path, workflow_name: str, ask: str | None) -> None:
    """Run the optional LLM agent over a finished comparison and echo its answer.

    Imported lazily so `validate` works without the 'llm' extra installed.
    """
    try:
        from theiavalidate.agent import run_agent
    except ImportError as err:
        raise click.ClickException(
            f"the --summarize/--ask agent needs the 'llm' extra: "
            f"pip install 'theiavalidate[llm]' ({err})"
        )

    click.echo("\nRunning agent...")
    try:
        state = run_agent(
            ask or _DEFAULT_QUESTION,
            result,
            criteria_yaml=str(criteria_path),
            workflow_name=workflow_name,
        )
    except Exception as err:
        click.secho(f"\nagent failed (skipping summary): {err}", fg="yellow")
        return

    click.secho("\n--- Agent summary ---", bold=True)
    click.echo(state.answer or "(the agent returned no text)")


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
@click.option(
    "--exit-zero",
    is_flag=True,
    help="Always exit 0, even when differences are found.",
)
@click.option(
    "--summarize",
    is_flag=True,
    help="After comparing, run the optional LLM agent to summarize the differences "
    "(needs the 'llm' extra and ANTHROPIC_API_KEY).",
)
@click.option(
    "--ask",
    help="Ask the LLM agent a specific question about the comparison (implies "
    "--summarize).",
)
def validate(
    table1: Path,
    table2: Path,
    config_path: Path | None,
    preset: str | None,
    key: str | None,
    key1: str | None,
    key2: str | None,
    outdir: Path,
    prefix: str,
    html: bool,
    pdf: bool,
    exit_zero: bool,
    summarize: bool,
    ask: str | None,
) -> None:
    """Compare TABLE1 and TABLE2 using a config (or preset)."""
    if bool(config_path) == bool(preset):
        raise click.UsageError("provide exactly one of --config or --preset")

    try:
        criteria_path, workflow_name = _criteria_source(config_path, preset)
        config = Config.from_yaml(str(criteria_path))
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

    if summarize or ask:
        _run_agent(result, criteria_path, workflow_name, ask)

    if result.passed:
        click.secho("PASSED, LGTM", fg="green", bold=True)
    else:
        click.secho("DIFFERENCES FOUND", fg="red", bold=True)
        if not exit_zero:
            sys.exit(1)


if __name__ == "__main__":
    main()
