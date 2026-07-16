"""CLI surface: version, config/preset selection, and exit codes."""

import textwrap

from click.testing import CliRunner

from theiavalidate.cli import main


def _write(path, text):
    path.write_text(textwrap.dedent(text).lstrip())


def _tables(tmp_path, right_len="100"):
    left = tmp_path / "left.tsv"
    right = tmp_path / "right.tsv"
    _write(left, "id\tlen\na\t100\n")
    _write(right, f"id\tlen\na\t{right_len}\n")
    return left, right


def _config(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    _write(
        cfg,
        """
        key: id
        columns:
          len:
            type: float
            method: percent_diff
            threshold: 0.01
        """,
    )
    return cfg


class TestVersion:
    def test_version_flag(self):
        result = CliRunner().invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "2.0.0.dev0" in result.output


class TestValidateSelection:
    def test_requires_exactly_one_of_config_or_preset(self, tmp_path):
        left, right = _tables(tmp_path)
        result = CliRunner().invoke(main, ["validate", str(left), str(right)])
        assert result.exit_code != 0
        assert "exactly one of --config or --preset" in result.output

    def test_config_and_preset_together_rejected(self, tmp_path):
        left, right = _tables(tmp_path)
        cfg = _config(tmp_path)
        result = CliRunner().invoke(
            main,
            ["validate", str(left), str(right), "--config", str(cfg), "--preset", "x"],
        )
        assert result.exit_code != 0
        assert "exactly one of --config or --preset" in result.output


class TestValidateExitCodes:
    def test_passing_run_exits_zero(self, tmp_path):
        left, right = _tables(tmp_path, right_len="100")
        cfg = _config(tmp_path)
        result = CliRunner().invoke(
            main,
            ["validate", str(left), str(right), "--config", str(cfg),
             "--outdir", str(tmp_path), "--no-html"],
        )
        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_differences_exit_nonzero(self, tmp_path):
        left, right = _tables(tmp_path, right_len="500")
        cfg = _config(tmp_path)
        result = CliRunner().invoke(
            main,
            ["validate", str(left), str(right), "--config", str(cfg),
             "--outdir", str(tmp_path), "--no-html"],
        )
        assert result.exit_code == 1
        assert "DIFFERENCES FOUND" in result.output

    def test_writes_summary_output(self, tmp_path):
        left, right = _tables(tmp_path)
        cfg = _config(tmp_path)
        CliRunner().invoke(
            main,
            ["validate", str(left), str(right), "--config", str(cfg),
             "--outdir", str(tmp_path), "--prefix", "run", "--no-html"],
        )
        assert (tmp_path / "run_summary.tsv").exists()
