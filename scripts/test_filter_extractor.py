"""
Quick test script for FilterExtractor using metadata_filled.csv, authority_map.csv, and acronym_map.csv.
Run with:
    python scripts/test_filter_extractor.py
"""

import pandas as pd
from pathlib import Path
from healthcare_rag_llm.filters.filter_extractor import FilterExtractor

# ---------------------------------------------------------------------
# 1️⃣ Paths
# ---------------------------------------------------------------------
DATA_DIR = Path("data/metadata")
meta_path = DATA_DIR / "metadata_filled.csv"
auth_path = DATA_DIR / "authority_map.csv"
acr_path = DATA_DIR / "acronym_map.csv"

# ---------------------------------------------------------------------
# 2️⃣ Load CSV files
# ---------------------------------------------------------------------
meta_df = pd.read_csv(meta_path)
auth_df = pd.read_csv(auth_path)
acr_df = pd.read_csv(acr_path)

print(f"✅ Loaded metadata: {len(meta_df)} docs, {len(auth_df)} authorities, {len(acr_df)} acronyms.\n")

# ---------------------------------------------------------------------
# 3️⃣ Build maps and metadata list
# ---------------------------------------------------------------------
authority_map = {
    str(row["authority_abbr"]).lower().strip(): str(row["authority_full"]).strip()
    for _, row in auth_df.iterrows()
    if pd.notna(row["authority_abbr"]) and pd.notna(row["authority_full"])
}

acronym_map = {
    str(row["acronym"]).lower().strip(): str(row["full_term"]).strip()
    for _, row in acr_df.iterrows()
    if pd.notna(row["acronym"]) and pd.notna(row["full_term"])
}

# metadata list
doc_metadata = meta_df.to_dict(orient="records")

# ---------------------------------------------------------------------
# 4️⃣ Initialize FilterExtractor
# ---------------------------------------------------------------------
extractor = FilterExtractor(
    authority_map=authority_map,
    acronym_map=acronym_map,
    doc_metadata=doc_metadata
)

print("✅ FilterExtractor initialized.\n")

# ---------------------------------------------------------------------
# 5️⃣ Test queries
# ---------------------------------------------------------------------
test_queries = [
    "What are the NYS DOH medicaid updates after 2019?",
    "Show Recipient Restriction Program pdf file from NYS DOH between Jan 2020 and May 2022.",
    "What SPA documents did NYS DOH issue before 2021?",
    "Medicaid updates from New York State Department of Health after March 2023"
]

for q in test_queries:
    print("---------------------------------------------------")
    print("Query:", q)
    filters = extractor.extract(q)
    print("Extracted filters:", filters)
    print()

print("✅ Test completed.")
