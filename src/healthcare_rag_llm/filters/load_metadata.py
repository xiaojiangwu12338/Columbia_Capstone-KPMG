# src/healthcare_rag_llm/filters/load_metadata.py

import pandas as pd
from pathlib import Path
from healthcare_rag_llm.filters.llm_filter_extractor import LLMFilterExtractor as FilterExtractor

DATA_DIR = Path("data/metadata")

def build_filter_extractor() -> FilterExtractor:
    """Load CSV files and build FilterExtractor."""
    # 1. Load metadata_filled.csv
    meta_df = pd.read_csv(DATA_DIR / "metadata_filled.csv")
    doc_metadata = meta_df.to_dict(orient="records")

    # 2. Load authority_map.csv
    auth_df = pd.read_csv(DATA_DIR / "authority_map.csv")
    authority_map = {
        str(row["authority_abbr"]).lower(): str(row["authority_full"]).strip()
        for _, row in auth_df.iterrows()
    }

    # 3. Load acronym_map.csv
    acr_df = pd.read_csv(DATA_DIR / "acronym_map.csv")
    acronym_map = {
        str(row["acronym"]).lower(): str(row["full_term"]).strip()
        for _, row in acr_df.iterrows()
    }

    # 4. Build and return extractor
    return FilterExtractor(
        authority_map=authority_map,
        acronym_map=acronym_map,
        doc_metadata=doc_metadata
    )
