import re
from datetime import date
from typing import Dict, List

# --------------------------------------------------------
# Month lookup table
# --------------------------------------------------------
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

# --------------------------------------------------------
# Date parsing helper
# --------------------------------------------------------
def _parse_date_expr(q: str) -> Dict[str, str]:
    """
    Parse natural language date expressions like:
    'after March 2023', 'before 2022', 'between Jan 2020 and Mar 2021'
    Returns dict with min_effective_date / max_effective_date if found.
    """
    q = q.lower()
    out = {}

    # --- between X and Y ---
    m_between = re.search(
        r'between\s+(?P<m1>[a-z]+)?\s*(?P<y1>\d{4})\s+and\s+(?P<m2>[a-z]+)?\s*(?P<y2>\d{4})',
        q
    )
    if m_between:
        m1, y1, m2, y2 = m_between.group("m1"), m_between.group("y1"), m_between.group("m2"), m_between.group("y2")
        mm1 = MONTHS.get(m1[:3], 1) if m1 else 1
        mm2 = MONTHS.get(m2[:3], 12) if m2 else 12
        out["min_effective_date"] = date(int(y1), mm1, 1).isoformat()
        out["max_effective_date"] = date(int(y2), mm2, 28).isoformat()
        return out

    # --- after / since / post ---
    m_after = re.search(r'(after|since|post)\s+([a-z]+)?\s*(\d{4})', q)
    if m_after:
        month = m_after.group(2)
        year = int(m_after.group(3))
        mm = MONTHS.get(month[:3], 1) if month else 1
        out["min_effective_date"] = date(year, mm, 1).isoformat()
        return out

    # --- before / until / prior / through ---
    m_before = re.search(r'(before|until|prior|through)\s+([a-z]+)?\s*(\d{4})', q)
    if m_before:
        month = m_before.group(2)
        year = int(m_before.group(3))
        mm = MONTHS.get(month[:3], 12) if month else 12
        out["max_effective_date"] = date(year, mm, 28).isoformat()
        return out

    # --- simple year only ---
    year_match = re.findall(r"(19|20)\d{2}", q)
    if year_match:
        year = max(int(y) for y in year_match)
        out["min_effective_date"] = f"{year}-01-01"

    return out


# --------------------------------------------------------
# Main FilterExtractor class
# --------------------------------------------------------
class FilterExtractor:
    """
    Extract structured filters (authority, doc_type, effective_date, etc.)
    from natural-language queries, based on metadata and mapping CSVs.
    """

    def __init__(self, authority_map: Dict[str, str], acronym_map: Dict[str, str], doc_metadata: List[Dict]):
        self.authority_map = {k.lower(): v for k, v in authority_map.items()}
        self.acronym_map = {k.lower(): v for k, v in acronym_map.items()}

        # --- metadata_filled.csv fields ---
        self.authorities = {
            str(d.get("authority_abbr", "")).lower(): str(d.get("authority_name", ""))
            for d in doc_metadata if d.get("authority_abbr")
        }
        self.doc_titles = [str(d.get("doc_title", "")).lower() for d in doc_metadata if d.get("doc_title")]
        self.doc_types = list({str(d.get("doc_type", "")).lower() for d in doc_metadata if d.get("doc_type")})

    # --------------------------------------------------------
    def extract(self, query: str) -> Dict:
        """Extract structured filters from a user query."""
        q = query.lower()
        filters = {}

        # 1. Authority detection
        authorities = []
        for abbr, full in self.authority_map.items():
            if abbr in q or full.lower() in q:
                authorities.append(full)
        for abbr, full in self.authorities.items():
            if abbr in q or full.lower() in q:
                authorities.append(full)
        if authorities:
            filters["authority_names"] = list(set(authorities))

        # 2. Program / acronym detection
        keywords = [v for k, v in self.acronym_map.items() if k.lower() in q]
        if keywords:
            filters["keywords"] = list(set(keywords))

        # 3. Document title match
        matched_titles = [t for t in self.doc_titles if t and t in q]
        if matched_titles:
            filters["doc_titles"] = matched_titles

        # 4. Document type detection
        matched_types = [t for t in self.doc_types if t and t in q]
        if matched_types:
            filters["doc_types"] = matched_types

        # 5. Date parsing (year, month, range)
        date_filters = _parse_date_expr(q)
        filters.update(date_filters)

        return filters