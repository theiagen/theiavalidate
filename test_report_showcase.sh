#!/usr/bin/env bash
# Report showcase: generate two example datasets and render both HTML reports.
#
#   fail  -> tests/report_showcase/out/showcase_fail_report.html
#            DIFFERENCES FOUND; n_differences counts span the full heat gradient
#            (green -> yellow -> red -> critical purple).
#   pass  -> tests/report_showcase/out/showcase_pass_report.html
#            PASSED; every column matches.
#
# Both datasets are compared with the SAME config, which exercises every
# comparison method: exact, ignore, percent_diff, range (numeric), range (date),
# file_exact, plus the any_of and all_of combinators.
set -euo pipefail

DIR="tests/report_showcase"
OUT="$DIR/out"

# 1. (Re)generate the fixtures and the files that file_exact hashes.
uv run --project . python "$DIR/make_fixtures.py"

# 2. Render the failing report (walks the heat gradient)
#    `|| true` keeps the script going since a diff exits non-zero.
uv run --project . theiavalidate validate \
    "$DIR/fail/left.tsv" "$DIR/fail/right.tsv" \
    --config "$DIR/showcase.yaml" \
    --outdir "$OUT" --prefix showcase_fail || true

# 3. Render the passing report.
uv run --project . theiavalidate validate \
    "$DIR/pass/left.tsv" "$DIR/pass/right.tsv" \
    --config "$DIR/showcase.yaml" \
    --outdir "$OUT" --prefix showcase_pass

echo
echo "Reports written to $OUT/"
echo "  fail: $OUT/showcase_fail_report.html"
echo "  pass: $OUT/showcase_pass_report.html"
