"""Generate the report-showcase fixtures.

Writes two dataset pairs plus the files that `file_exact` hashes:

    pass/left.tsv  pass/right.tsv   -> every column matches (PASSED report)
    fail/left.tsv  fail/right.tsv   -> engineered differences (DIFFERENCES report)

The fail set is tuned so the per-column ``n_differences`` counts walk the whole
report heat gradient. With 12 compared rows the counts are:

    column         diffs   heat
    exact_pass       0     green  (no differences)
    ignore_me        0     green  (values differ, but method is `ignore`)
    range_date       1     yellow (fewest)
    pct_float        2       .
    range_int        4       .
    exact_str        6       . ramp yellow -> red
    file_hash        8       .
    any_of_float    10       .
    all_of_float    11     red    (most, of the non-critical columns)
    set_field       12     purple (every compared row differs -> critical)

so a single failing report shows green, the full yellow->red ramp, and the
critical purple all at once. The fail set also carries an extra row and extra
unconfigured columns to populate the report's "What didn't line up" section.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FILES = HERE / "files"
N = 12  # compared rows: sample01 .. sample12
SAMPLES = [f"sample{i:02d}" for i in range(1, N + 1)]


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _file_columns() -> tuple[list[str], list[str], list[str]]:
    """Materialise the files `file_exact` reads and return absolute-path columns.

    fail: rows 1-8 point at differing content, rows 9-12 at identical content.
    pass: both sides point at the same shared file, so every row matches.
    Absolute paths keep the fixtures resolvable no matter the working directory.
    """
    fail_left, fail_right, shared = [], [], []
    for i, sample in enumerate(SAMPLES, start=1):
        if i <= 8:  # differing content -> file_exact mismatch
            left = _write_file(FILES / "fail" / "left" / f"{sample}.txt", f"left-{sample}")
            right = _write_file(FILES / "fail" / "right" / f"{sample}.txt", f"right-{sample}")
        else:  # identical content -> match
            same = f"identical-{sample}"
            left = _write_file(FILES / "fail" / "left" / f"{sample}.txt", same)
            right = _write_file(FILES / "fail" / "right" / f"{sample}.txt", same)
        fail_left.append(str(left))
        fail_right.append(str(right))
        # pass: one shared file both sides reference
        shared_path = _write_file(FILES / "pass" / f"{sample}.txt", f"shared-{sample}")
        shared.append(str(shared_path))
    return fail_left, fail_right, shared


def _rng(i: int, breakpoint: int, differ, same):
    """Value for row i: `differ` for i <= breakpoint, else `same`."""
    return differ(i) if i <= breakpoint else same(i)


def build_fail(fail_left_files, fail_right_files) -> tuple[pd.DataFrame, pd.DataFrame]:
    left_rows, right_rows = [], []
    for i, sample in enumerate(SAMPLES, start=1):
        # any_of / all_of value tables (indexed 1-based); paired L/R per row.
        any_l = "100"
        any_r = {  # 10 fail (both branches miss), 2 pass
            1: "200", 2: "150", 3: "175", 4: "160", 5: "130",
            6: "120", 7: "115", 8: "110", 9: "108", 10: "106",
            11: "102",  # within absolute 3 -> range branch passes
            12: "100",  # identical
        }[i]
        all_l = "1000"
        all_r = {  # 11 fail (one branch missed), 1 pass
            1: "1150", 2: "1200", 3: "1300", 4: "300", 5: "250",
            6: "1500", 7: "2000", 8: "400", 9: "1101", 10: "1102",
            11: "1105",
            12: "1050",  # within 50% AND within 100 -> passes
        }[i]

        left_rows.append({
            "samplename": sample,
            "exact_pass": f"val-{sample}",                 # 0 diffs
            "exact_str": f"alpha-{sample}" if i <= 6 else f"gamma-{sample}",
            "ignore_me": f"left-note-{sample}",            # differs, ignored
            "pct_float": "100",
            "range_int": "10",
            "range_date": "2024-01-01" if i == 1 else "2024-03-15",
            "file_hash": fail_left_files[i - 1],
            "any_of_float": any_l,
            "all_of_float": all_l,
            "set_field": f"a{i:02d},b",
            "lab_notes": f"note-{sample}",                 # unconfigured, left only
        })
        right_rows.append({
            "samplename": sample,
            "exact_pass": f"val-{sample}",
            "exact_str": f"beta-{sample}" if i <= 6 else f"gamma-{sample}",  # 6 diffs
            "ignore_me": f"right-note-{sample}",
            "pct_float": _rng(i, 2, lambda i: {1: "50", 2: "200"}[i], lambda i: "100"),  # 2 diffs
            "range_int": _rng(i, 4, lambda i: {1: "20", 2: "15", 3: "14", 4: "100"}[i], lambda i: "10"),  # 4 diffs
            "range_date": "2024-01-10" if i == 1 else "2024-03-15",  # 1 diff (9 days)
            "file_hash": fail_right_files[i - 1],          # 8 diffs
            "any_of_float": any_r,                         # 10 diffs
            "all_of_float": all_r,                         # 11 diffs
            "set_field": f"a{i:02d},c",                    # 12 diffs -> critical
            "qc_flag": "PASS",                             # unconfigured, right only
        })

    # An extra row only in the right table -> populates "Rows only in ...".
    right_rows.append({
        "samplename": "sample13",
        "exact_pass": "val-sample13", "exact_str": "gamma-sample13",
        "ignore_me": "right-note-sample13", "pct_float": "100", "range_int": "10",
        "range_date": "2024-03-15", "file_hash": fail_right_files[0],
        "any_of_float": "100", "all_of_float": "1000", "set_field": "a13,c",
        "qc_flag": "PASS",
    })
    return pd.DataFrame(left_rows), pd.DataFrame(right_rows)


def build_pass(shared_files) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, sample in enumerate(SAMPLES, start=1):
        rows.append({
            "samplename": sample,
            "exact_pass": f"val-{sample}",
            "exact_str": f"gamma-{sample}",
            "ignore_me": f"note-{sample}",
            "pct_float": "100",
            "range_int": "10",
            "range_date": "2024-03-15",
            "file_hash": shared_files[i - 1],  # both sides reference the same file
            "any_of_float": "100",
            "all_of_float": "1000",
            "set_field": "a,b",
        })
    df = pd.DataFrame(rows)
    return df.copy(), df.copy()


def main() -> None:
    fail_left_files, fail_right_files, shared_files = _file_columns()

    fail_left, fail_right = build_fail(fail_left_files, fail_right_files)
    (HERE / "fail").mkdir(exist_ok=True)
    fail_left.to_csv(HERE / "fail" / "left.tsv", sep="\t", index=False)
    fail_right.to_csv(HERE / "fail" / "right.tsv", sep="\t", index=False)

    pass_left, pass_right = build_pass(shared_files)
    (HERE / "pass").mkdir(exist_ok=True)
    pass_left.to_csv(HERE / "pass" / "left.tsv", sep="\t", index=False)
    pass_right.to_csv(HERE / "pass" / "right.tsv", sep="\t", index=False)

    print(f"wrote fixtures under {HERE}")


if __name__ == "__main__":
    main()
