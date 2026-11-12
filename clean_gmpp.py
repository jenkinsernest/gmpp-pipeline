#!/usr/bin/env python3
"""Clean a raw GMPP annual CSV into the 10-column layout used by train_models.py.

There was no standalone cleaning script in the GMPP folder. The rules below
were reconstructed by comparing each raw file with its cleaned companion
(checked 12 November 2026).

Keep a row if the IPA Delivery Confidence Assessment is Red, Amber or Green.
If that IPA cell is genuinely empty, fall back to the SRO rating when the SRO
cell is Red, Amber or Green. If the IPA cell has any other text (FOI wording,
"Not Applicable", the literal "NA", planning text), drop the row even if the
SRO cell has a rating.

Column titles are matched by pattern, so the same cleaner works for all five
known layouts, including the 2024-25 redesign and the 2025-26 title changes.
Values are copied as published. Money, dates and rating case are not rewritten
here; feature_engineering.py handles that during training.

This module never overwrites a file unless the caller passes an output path.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from known_layouts import CLEANED_COLUMNS, match_known_layout, normalize_header

RAG = {"red", "amber", "green"}

# pandas would treat the text "NA" as missing. In the 2022-23 file "NA" is a
# published label meaning no IPA rating was given, and the historical cleaner
# did not fall back to the SRO rating in that case. Read rating-related fields
# as plain text.
CSV_READ = dict(dtype=str, keep_default_na=False, encoding="utf-8-sig")


def is_blank(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "<na>", "none", "null"}


def is_rag(value) -> bool:
    if is_blank(value):
        return False
    return str(value).strip().casefold() in RAG


def tidy_cell(value) -> str:
    """Strip ends and collapse line breaks, matching the stored cleaned files."""
    text = str(value).replace(chr(13)+chr(10), " ").replace(chr(13), " ").replace(chr(10), " ")
    return " ".join(text.split())


def pick_dca(ipa, sro):
    if is_rag(ipa):
        return str(ipa).strip()
    if is_blank(ipa) and is_rag(sro):
        return str(sro).strip()
    return None


def _norm_cols(df: pd.DataFrame) -> dict[str, str]:
    return {normalize_header(c): c for c in df.columns}


def find_column(df: pd.DataFrame, predicate, required=True, label=""):
    matches = [c for c in df.columns if predicate(normalize_header(c))]
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    raise KeyError(f"Could not uniquely find {label or 'column'}: {matches}")


def map_raw_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Map cleaned column names to raw column names (Benefits may be missing)."""

    def title_is(*exact):
        targets = {normalize_header(x) for x in exact}
        return lambda n: n in targets

    def starts(*prefixes):
        prefs = tuple(normalize_header(p) for p in prefixes)
        return lambda n: n.startswith(prefs) or n in prefs

    def is_id(n):
        return n in {
            "GMPP ID Number",
            "GMPP ID",
            "Major Projects ID",
        } or n.rstrip(":") in {"GMPP ID Number", "GMPP ID", "Major Projects ID"}

    def is_benefits(n):
        # Do not pick the narrative column.
        if "narrative" in n.casefold():
            return False
        return n == "TOTAL Baseline Benefits (£m)" or n.startswith("Benefits (£m")

    def is_start(n):
        if "narrative" in n.casefold():
            return False
        return n == "Start Date" or n.startswith("Project - Start Date")

    def is_end(n):
        if "narrative" in n.casefold():
            return False
        return n == "End Date" or n.startswith("Project - End Date")

    def is_wlc(n):
        if "narrative" in n.casefold():
            return False
        return n.startswith("TOTAL Baseline Whole Life Costs") or n.startswith("Whole Life Cost")

    def is_baseline(n):
        return n.startswith("Financial Year Baseline")

    def is_forecast(n):
        return n.startswith("Financial Year Forecast")

    def is_ipa(n):
        return n.startswith("IPA Delivery Confidence Assessment")

    def is_sro(n):
        return n.startswith("SRO Delivery Confidence Assessment")

    mapping = {
        "Major Projects ID": find_column(df, is_id, label="project ID"),
        "Department": find_column(df, title_is("Department"), label="Department"),
        "Annual Report Category": find_column(
            df, title_is("Annual Report Category"), label="Annual Report Category"
        ),
        "Whole Life Cost": find_column(df, is_wlc, label="Whole Life Cost"),
        "Financial Year Baseline": find_column(df, is_baseline, label="Financial Year Baseline"),
        "Financial Year Forecast": find_column(df, is_forecast, label="Financial Year Forecast"),
        "Benefits": find_column(df, is_benefits, required=False, label="Benefits"),
        "Start Date": find_column(df, is_start, label="Start Date"),
        "End Date": find_column(df, is_end, label="End Date"),
        "IPA_SRC": find_column(df, is_ipa, label="IPA Delivery Confidence Assessment"),
        "SRO_SRC": find_column(df, is_sro, label="SRO Delivery Confidence Assessment"),
    }
    return mapping


def read_raw_spreadsheet(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **CSV_READ)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, dtype=str, engine="openpyxl").fillna("")
    if suffix == ".xls":
        return pd.read_excel(path, dtype=str).fillna("")
    if suffix == ".ods":
        return pd.read_excel(path, dtype=str, engine="odf").fillna("")
    raise ValueError(f"Unsupported spreadsheet type: {path.suffix}")


def clean_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mapping = map_raw_columns(raw)
    ipa_col = mapping["IPA_SRC"]
    sro_col = mapping["SRO_SRC"]
    rows = []
    dropped = 0
    used_sro = 0
    for rec in raw.to_dict(orient="records"):
        dca = pick_dca(rec[ipa_col], rec[sro_col])
        if dca is None:
            dropped += 1
            continue
        if is_blank(rec[ipa_col]) and is_rag(rec[sro_col]):
            used_sro += 1
        out = {}
        for dest in CLEANED_COLUMNS:
            if dest == "IPA Delivery Confidence":
                out[dest] = dca
                continue
            src = mapping.get(dest)
            if not src:
                out[dest] = ""
                continue
            value = rec.get(src, "")
            out[dest] = "" if is_blank(value) else tidy_cell(value)
        rows.append(out)
    cleaned = pd.DataFrame(rows, columns=CLEANED_COLUMNS)
    stats = {
        "raw_rows": int(len(raw)),
        "cleaned_rows": int(len(cleaned)),
        "dropped_rows": int(dropped),
        "used_sro_fallback": int(used_sro),
        "column_map": {k: v for k, v in mapping.items()},
    }
    return cleaned, stats


def clean_file(path: Path, dest: Path | None = None) -> tuple[pd.DataFrame, dict]:
    raw = read_raw_spreadsheet(path)
    layout_status, layout_label, layout_detail = match_known_layout(list(raw.columns))
    cleaned, stats = clean_frame(raw)
    stats.update({
        "source": str(path),
        "layout_status": layout_status,
        "layout_label": layout_label,
        "layout_detail": layout_detail,
        "raw_columns": list(raw.columns),
    })
    if dest is not None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(dest, index=False, encoding="utf-8-sig")
        stats["output"] = str(dest)
    return cleaned, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Clean a raw GMPP annual spreadsheet.")
    parser.add_argument("source", type=Path, help="Raw CSV, XLSX, XLS or ODS file")
    parser.add_argument("-o", "--output", type=Path, help="Where to write cleaned-gmpp-YYYY-YY.csv")
    args = parser.parse_args(argv)
    cleaned, stats = clean_file(args.source, args.output)
    print(
        f"Cleaned {stats['cleaned_rows']} of {stats['raw_rows']} rows "
        f"(dropped {stats['dropped_rows']}; SRO fallback {stats['used_sro_fallback']})."
    )
    print(f"Layout: {stats['layout_status']} {stats['layout_label'] or ''}".strip())
    if args.output:
        print(f"Wrote {args.output}")
    else:
        print("No output path given, so nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
