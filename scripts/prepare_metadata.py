"""
scripts/prepare_metadata.py

Automatically fill effective_date and doc_type for metadata.
Usage:
    python scripts/prepare_metadata.py
"""

import pandas as pd
import re
from datetime import date
from pathlib import Path

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

def infer_effective_date(text: str) -> str | None:
    """
    Try to extract date from filename or document title.
    e.g. "mu_no03_feb19_pr.pdf" → 2019-02-01
         "New York State Medicaid Update January 2021" → 2021-01-01
    """
    if not isinstance(text, str):
        return None

    text = text.lower()

    # pattern 1: filename style "feb19"
    m1 = re.search(r'_(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{2})_', text)
    if m1:
        mm = MONTHS[m1.group(1)]
        yy = 2000 + int(m1.group(2))
        return date(yy, mm, 1).isoformat()

    # pattern 2: text style "February 2019"
    m2 = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})', text)
    if m2:
        mm = MONTHS[m2.group(1)]
        yy = int(m2.group(2))
        return date(yy, mm, 1).isoformat()

    return None


def prepare_metadata(csv_path="data/metadata/metadata.csv"):
    """
    Load metadata CSV, fill effective_date & doc_type if missing,
    and save to a new versioned file.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    for col in ["file_name", "doc_title"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Fill doc_type automatically
    if "doc_type" not in df.columns:
        df["doc_type"] = None
    df["doc_type"] = df["file_name"].apply(lambda x: Path(x).suffix.replace(".", "") if isinstance(x, str) else None)

    # Fill effective_date
    if "effective_date" not in df.columns:
        df["effective_date"] = None

    for i, row in df.iterrows():
        if not row["effective_date"] or pd.isna(row["effective_date"]):
            guessed = infer_effective_date(row["file_name"]) or infer_effective_date(row["doc_title"])
            if guessed:
                df.at[i, "effective_date"] = guessed

    # Save updated version
    output_path = csv_path.parent / f"{csv_path.stem}_filled.csv"
    df.to_csv(output_path, index=False)
    print(f"[✅] Metadata updated and saved to: {output_path}")


if __name__ == "__main__":
    prepare_metadata()