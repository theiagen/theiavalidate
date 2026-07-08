"""Render a ComparisonResult to a self-contained HTML report, and optionally PDF.

HTML is built with pandas + stdlib only (no extra dependencies), so it always
works. PDF conversion uses `pdfkit` (a core dependency) and needs the
`wkhtmltopdf` system binary; it converts the same HTML document.
"""

from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import pdfkit

if TYPE_CHECKING:
    from theiavalidate.results import ComparisonResult

_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: #1a1a1a; margin: 24px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.subtitle { color: #666; margin: 0 0 16px; font-size: 13px; }
.banner { display: inline-block; padding: 6px 14px; border-radius: 4px;
          font-weight: 600; font-size: 14px; margin-bottom: 20px; }
.banner.pass { background: #d4edda; color: #155724; }
.banner.fail { background: #f8d7da; color: #721c24; }
h2 { font-size: 15px; border-bottom: 1px solid #ddd; padding-bottom: 4px;
     margin: 24px 0 10px; }
.scroll { overflow-x: auto; }
table.tv-table { border-collapse: collapse; font-size: 12px; margin-bottom: 8px; }
table.tv-table th, table.tv-table td { border: 1px solid #ccc; padding: 4px 8px;
                                       text-align: left; white-space: nowrap; }
table.tv-table th { background: #f0f0f0; }
ul.legend { font-size: 12px; color: #444; padding-left: 18px; }
.muted { color: #888; font-size: 12px; }
"""


def render(
    result: "ComparisonResult",
    outdir: str,
    *,
    prefix: str = "theiavalidate",
    html: bool = True,
    pdf: bool = False,
) -> list[Path]:
    """Write the report. Returns the paths written (html and/or pdf)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    document = _document(result)

    written: list[Path] = []
    if html:
        path = out / f"{prefix}_report.html"
        path.write_text(document, encoding="utf-8")
        written.append(path)
    if pdf:
        written.append(_write_pdf(document, out / f"{prefix}_report.pdf"))
    return written


def _document(result: "ComparisonResult") -> str:
    title = f"{result.left_name} vs {result.right_name}"
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{_esc(title)}</h1>"
        f"<p class='subtitle'>Validation report — {date.today().isoformat()}</p>"
        f"{_banner(result.passed)}"
        f"{_summary_section(result)}"
        f"{_differences_section(result)}"
        f"{_exclusives_section(result)}"
        f"{_legend()}"
        "</body></html>"
    )


def _banner(passed: bool) -> str:
    cls, text = ("pass", "PASSED") if passed else ("fail", "DIFFERENCES FOUND")
    return f"<div class='banner {cls}'>{text}</div>"


def _summary_section(result: "ComparisonResult") -> str:
    summary = result.summary_df()
    if summary.empty:
        body = "<p class='muted'>No columns compared.</p>"
    else:
        body = _table(summary.to_html(classes="tv-table", border=0, na_rep=""))
    return f"<h2>Summary</h2>{body}"


def _differences_section(result: "ComparisonResult") -> str:
    diffs = result.differences_df()
    if diffs.empty:
        body = "<p class='muted'>No differences found.</p>"
    else:
        body = _table(diffs.to_html(classes="tv-table", border=0, na_rep=""))
    return f"<h2>Differences</h2>{body}"


def _exclusives_section(result: "ComparisonResult") -> str:
    left, right = result.left_name, result.right_name
    missing = (
        ", ".join(
            f"{_esc(col)} ({_esc(where)})"
            for col, where in result.missing_columns.items()
        )
        or "<span class='muted'>none</span>"
    )
    return (
        "<h2>What didn't line up</h2>"
        f"<p><b>Configured columns missing:</b> {missing}</p>"
        f"<p><b>Rows only in {_esc(left)}:</b> {_items(result.rows_only_left)}</p>"
        f"<p><b>Rows only in {_esc(right)}:</b> {_items(result.rows_only_right)}</p>"
        f"<p><b>Columns only in {_esc(left)} (not compared):</b> "
        f"{_items(result.columns_only_left)}</p>"
        f"<p><b>Columns only in {_esc(right)} (not compared):</b> "
        f"{_items(result.columns_only_right)}</p>"
    )


def _legend() -> str:
    return (
        "<h2>Methods</h2><ul class='legend'>"
        "<li><b>exact</b> — values (or sets/lists) must be equal</li>"
        "<li><b>ignore</b> — column skipped (always passes)</li>"
        "<li><b>percent_diff</b> — within a fractional tolerance of each other</li>"
        "<li><b>range</b> — within an absolute tolerance (days, for dates)</li>"
        "<li><b>file_exact</b> — referenced files are byte-identical (md5)</li>"
        "</ul>"
    )


def _table(inner: str) -> str:
    return f"<div class='scroll'>{inner}</div>"


def _items(values) -> str:
    if not len(values):
        return "<span class='muted'>none</span>"
    return ", ".join(_esc(str(v)) for v in values)


def _esc(text: str) -> str:
    return escape(str(text))


def _write_pdf(document: str, path: Path) -> Path:

    options = {
        "page-size": "Letter",
        "orientation": "Landscape",
        "encoding": "UTF-8",
        "margin-top": "0.25in",
        "margin-right": "0.25in",
        "margin-bottom": "0.25in",
        "margin-left": "0.25in",
    }
    pdfkit.from_string(document, str(path), options=options)
    return path
