"""Render a ComparisonResult to a self-contained HTML report, and optionally PDF.

HTML is built with pandas + stdlib only (no extra dependencies), so it always
works. PDF conversion uses `pdfkit` (a core dependency) and needs the
`wkhtmltopdf` system binary; it converts the same HTML document.
"""

from __future__ import annotations

import base64
from datetime import date
from html import escape
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import pdfkit

if TYPE_CHECKING:
    from theiavalidate.results import ComparisonResult

# Theiagen brand palette, matching the docs "light" color scheme (extra.css):
#   blue  #116eb7  primary
#   green #1da74a  accent
#   ink   #262626  body text
# Colors are hardcoded (no CSS variables) so wkhtmltopdf's old WebKit renders
# the PDF identically to the HTML.
_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: #262626; margin: 0; line-height: 1.5; font-size: 14px; }
.header { background: #116eb7; padding: 16px 32px;
          border-bottom: 2px solid #1da74a; }
.header img.logo { height: 38px; display: block; }
main { padding: 24px 32px 32px; }
h1 { font-size: 22px; line-height: 1.25; margin: 0 0 2px; color: #116eb7; }
.subtitle { color: #595959; margin: 0 0 18px; font-size: 13px; }
.banner { display: inline-block; padding: 7px 16px; border-radius: 4px;
          font-weight: 700; font-size: 12px; letter-spacing: 0.05em;
          text-transform: uppercase; margin-bottom: 24px; }
.banner.pass { background: #d2f0dc; color: #14713a; border: 1px solid #1da74a; }
.banner.fail { background: #f8d7da; color: #721c24; border: 1px solid #dda2a8; }
h2 { font-size: 16px; color: #116eb7; border-bottom: 1px solid #1da74a;
     padding-bottom: 5px; margin: 30px 0 12px; }
.scroll { overflow-x: auto; margin-bottom: 10px; }
table.tv-table { border-collapse: collapse; font-size: 12px; width: 100%; }
table.tv-table th, table.tv-table td { border: 1px solid #ececec; padding: 5px 10px;
                                       text-align: left; }
table.tv-table th { background: #eaf2f9; color: #0d5a97; font-weight: 600;
                    border-bottom: 1px solid #116eb7;
                    position: sticky; top: 0; z-index: 2; }
table.tv-table tr:nth-child(even) td { background: #f4f8fb; }

/* Summary: natural full-width layout. Only the column-header row is resizable —
   drag a header's right edge — and cells wrap once a column is dragged narrow.
   (The row-label cells are also <th>, so the resize handle is scoped to the
   header row to keep it off every data row.) Resize is an interactive-only
   affordance; the PDF just renders the columns as laid out. */
table.tv-summary thead tr:first-child th { resize: horizontal; overflow: auto; }
table.tv-summary th, table.tv-summary td { white-space: normal;
                                           overflow-wrap: anywhere; }

/* Differences: a tidy 'one differing cell per row' table. Fixed layout keeps it
   to the page width — long values wrap in the two value columns rather than
   scrolling sideways — and the sample column is frozen so it stays visible if a
   row is wide enough to scroll. (Sticky is ignored by the PDF renderer, which is
   fine: the PDF just lays the table out statically.) */
table.tv-diff { table-layout: fixed; width: 100%; }
table.tv-diff th, table.tv-diff td { white-space: normal; word-break: break-word;
                                     vertical-align: top; }
table.tv-diff tr > *:nth-child(1) { width: 150px; }
table.tv-diff tr > *:nth-child(2) { width: 170px; color: #0d5a97; }
table.tv-diff tr > *:nth-child(3) { width: 90px; }
table.tv-diff tr > *:nth-child(6) { width: 70px; text-align: right; }
table.tv-diff tr > td:nth-child(1), table.tv-diff tr > th:nth-child(1) {
    position: sticky; left: 0; background: #eaf2f9; font-weight: 600; z-index: 1; }
table.tv-diff th:nth-child(1) { z-index: 3; }
/* Prose is capped for readability; wide tables above are not. */
.exclusives { max-width: 900px; }
.exclusives p { margin: 6px 0; }
.exclusives .label { color: #0d5a97; font-weight: 600; }
ul.legend { font-size: 13px; color: #333; padding-left: 20px; max-width: 900px;
            line-height: 1.7; }
.muted { color: #6b6b6b; }
.footer { margin-top: 32px; padding-top: 14px; border-top: 1px solid #e0e1e1;
          color: #6b6b6b; font-size: 11px; }
.footer img.symbol { height: 22px; vertical-align: middle; margin-right: 8px; }
"""


def _asset_data_uri(name: str) -> str:
    """Base64-embed a packaged image so the report stays self-contained."""
    data = (resources.files("theiavalidate") / "assets" / name).read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


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
    symbol = _asset_data_uri("theiagen-symbol.png")
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title>"
        f"<link rel='icon' type='image/png' href='{symbol}'>"
        f"<style>{_CSS}</style></head><body>"
        "<div class='header'>"
        f"<img class='logo' src='{_asset_data_uri('theiagen-logo-white.png')}'"
        " alt='Theiagen Genomics'></div>"
        "<main>"
        f"<h1>{_esc(title)}</h1>"
        f"<p class='subtitle'>Validation report — {date.today().isoformat()}</p>"
        f"{_banner(result.passed)}"
        f"{_summary_section(result)}"
        f"{_differences_section(result)}"
        f"{_exclusives_section(result)}"
        f"{_legend()}"
        "<p class='footer'>"
        f"<img class='symbol' src='{symbol}' alt=''>"
        "Generated by theiavalidate · Theiagen Genomics</p>"
        "</main></body></html>"
    )


def _banner(passed: bool) -> str:
    cls, text = ("pass", "PASSED") if passed else ("fail", "DIFFERENCES FOUND")
    return f"<div class='banner {cls}'>{text}</div>"


def _summary_section(result: "ComparisonResult") -> str:
    summary = result.summary_df()
    if summary.empty:
        body = "<p class='muted'>No columns compared.</p>"
    else:
        body = _table(
            summary.to_html(classes="tv-table tv-summary", border=0, na_rep="")
        )
    return f"<h2>Summary</h2>{body}"


def _differences_section(result: "ComparisonResult") -> str:
    diffs = result.differences_long_df()
    if diffs.empty:
        body = "<p class='muted'>No differences found.</p>"
    else:
        body = _table(
            diffs.to_html(
                classes="tv-table tv-diff", border=0, na_rep="", index=False
            )
        )
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
        "<h2>What didn't line up</h2><div class='exclusives'>"
        f"<p><span class='label'>Configured columns missing:</span> {missing}</p>"
        f"<p><span class='label'>Rows only in {_esc(left)}:</span> "
        f"{_items(result.rows_only_left)}</p>"
        f"<p><span class='label'>Rows only in {_esc(right)}:</span> "
        f"{_items(result.rows_only_right)}</p>"
        f"<p><span class='label'>Columns only in {_esc(left)} "
        f"(not compared):</span> {_items(result.columns_only_left)}</p>"
        f"<p><span class='label'>Columns only in {_esc(right)} "
        f"(not compared):</span> {_items(result.columns_only_right)}</p>"
        "</div>"
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
