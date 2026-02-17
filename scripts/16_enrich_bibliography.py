#!/usr/bin/env python3
"""
16_enrich_bibliography.py
=========================
Reads Scopus CSV export files, converts relevant entries to BibTeX,
deduplicates against the existing references.bib, and appends new entries.

Usage:
    python scripts/16_enrich_bibliography.py
"""

import csv
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("/Users/cristianespinal/Documents/magdalena_medio_research")

CSV_FILES = {
    "Ecuacion 1": BASE_DIR / "bib" / "Ecuación 1" / "ecuacion-1.csv",
    "Ecuacion 2": (
        BASE_DIR / "bib" / "Ecuación 2"
        / "scopus_export_Feb 14-2026_ca6af100-d50a-4368-8b4d-8fc692fa7084.csv"
    ),
    "Ecuacion 3": (
        BASE_DIR / "bib" / "Ecuación 3"
        / "scopus_export_Feb 15-2026_9e3da09b-5bb2-41f7-85ec-6eedc3e39ee7.csv"
    ),
}

BIB_FILE = BASE_DIR / "overleaf" / "references.bib"

# Terms for relevance filtering (case-insensitive).
# An entry is relevant if ANY term matches title, abstract, or author keywords.
RELEVANCE_TERMS = [
    # Deforestation / forest
    "deforestation", "forest loss", "forest cover change", "forest degradation",
    # Land use
    "land use change", "lulc", "land cover", "land use",
    # Conflict / peace
    "post-conflict", "peacebuilding", "armed conflict", "peace agreement",
    # Ecosystem services
    "ecosystem services", "carbon stock", "biodiversity", "carbon storage",
    # Remote sensing
    "remote sensing", "gee", "google earth engine", "landsat", "sentinel",
    # CA-Markov / simulation
    "ca-markov", "cellular automata", "land change model", "markov chain",
    # Spatial regression
    "gwr", "geographically weighted", "spatial regression",
    # Spatial statistics
    "hotspot analysis", "getis-ord", "spatial statistics", "moran",
    # Geography
    "colombia", "tropical deforestation", "amazon", "magdalena",
    # Accuracy / area estimation
    "olofsson", "accuracy assessment", "area estimation",
    # Hansen GFC
    "hansen", "global forest change",
    # InVEST
    "invest", "habitat quality", "water yield",
    # Machine learning
    "random forest", "machine learning classification",
]

# Compile a single regex alternation for speed
_RELEVANCE_RE = re.compile(
    "|".join(re.escape(t) for t in RELEVANCE_TERMS),
    re.IGNORECASE,
)

# Document types to exclude (unless cited_by > 20)
EXCLUDED_DOC_TYPES = {"review", "conference paper"}

# Stop words excluded from the "first significant title word" in citation keys
_KEY_STOP_WORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "to", "from",
    "with", "by", "at", "is", "are", "was", "were", "be", "been",
    "its", "their", "this", "that", "these", "those", "or", "as",
    "do", "does", "did", "has", "have", "had", "not", "no", "but",
    "can", "could", "will", "would", "may", "might", "shall", "should",
    "how", "what", "which", "who", "where", "when", "why",
    "using", "towards", "toward", "between", "through", "into",
    "about", "over", "under", "after", "before", "during", "within",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    """Remove diacritical marks, keeping base ASCII letters."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_col(name: str) -> str:
    """Lowercase and strip whitespace/BOM from a column header."""
    return name.strip().lower().strip("\ufeff")


def _get(row: dict, *candidates: str, default: str = "") -> str:
    """Return the first non-empty value for any of the candidate column names."""
    for key in candidates:
        val = row.get(key, "").strip()
        if val:
            return val
    return default


def _parse_cited_by(raw: str) -> int:
    """Safely parse citation count."""
    try:
        return int(raw.strip())
    except (ValueError, AttributeError):
        return 0


def _clean_author_bibtex(authors_str: str) -> str:
    """
    Convert Scopus author string to BibTeX author field.

    Scopus format:  "LastName1, F.N.; LastName2, F.N.; ..."
    BibTeX format:  "LastName1, F.N. and LastName2, F.N. and ..."
    """
    if not authors_str:
        return ""
    # Split on semicolons (Scopus delimiter between authors)
    parts = [a.strip() for a in authors_str.split(";") if a.strip()]
    return " and ".join(parts)


def _first_author_last_name(authors_str: str) -> str:
    """
    Extract the last name of the first author from Scopus "Authors" column.
    Scopus gives "LastName, F.N.; ...".
    Returns an ASCII-safe, CamelCase version suitable for a cite key.
    """
    if not authors_str:
        return "Unknown"
    first = authors_str.split(";")[0].strip()
    # The last name is everything before the first comma
    last = first.split(",")[0].strip()
    # Remove accents and non-alpha chars, then CamelCase
    last = _strip_accents(last)
    # Handle hyphenated or multi-word last names: keep them joined
    parts = re.split(r"[\s\-]+", last)
    key = "".join(p.capitalize() for p in parts if p)
    # Remove any remaining non-alphanumeric
    key = re.sub(r"[^A-Za-z]", "", key)
    return key or "Unknown"


def _first_significant_title_word(title: str) -> str:
    """
    Return the first 'significant' word in the title (skip stop words).
    Capitalised, ASCII-safe, alphanumeric only.
    """
    if not title:
        return "Untitled"
    words = re.findall(r"[A-Za-z]+", _strip_accents(title))
    for w in words:
        if w.lower() not in _KEY_STOP_WORDS:
            return w.capitalize()
    # Fallback: use the very first word
    return words[0].capitalize() if words else "Untitled"


def _make_citation_key(authors_str: str, year: str, title: str) -> str:
    """Generate citation key: FirstAuthorLastName + Year + FirstSignificantTitleWord."""
    author_part = _first_author_last_name(authors_str)
    year_part = year.strip() if year else "XXXX"
    title_part = _first_significant_title_word(title)
    return f"{author_part}{year_part}{title_part}"


def _escape_bibtex(text: str) -> str:
    """Minimal escaping for BibTeX string values."""
    # Replace & with \& (unless already escaped)
    text = re.sub(r"(?<!\\)&", r"\\&", text)
    return text


def _map_document_type(doc_type: str) -> str:
    """Map Scopus Document Type to a BibTeX entry type."""
    dt = doc_type.strip().lower()
    mapping = {
        "article": "article",
        "article in press": "article",
        "review": "article",
        "conference paper": "inproceedings",
        "book chapter": "incollection",
        "book": "book",
        "editorial": "article",
        "letter": "article",
        "note": "article",
        "short survey": "article",
        "erratum": "article",
        "data paper": "article",
    }
    return mapping.get(dt, "article")


def _build_pages(page_start: str, page_end: str, art_no: str) -> str:
    """Build a pages string from start/end or article number."""
    if page_start and page_end and page_start != page_end:
        return f"{page_start}--{page_end}"
    if page_start:
        return page_start
    if art_no:
        return art_no
    return ""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def read_scopus_csv(filepath: Path) -> list[dict]:
    """
    Read a Scopus CSV export and return a list of normalised row dicts.
    Handles BOM via utf-8-sig encoding.
    """
    rows: list[dict] = []
    with open(filepath, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw_row in reader:
            # Normalise column names
            row = {_normalize_col(k): v for k, v in raw_row.items()}
            rows.append(row)
    return rows


def is_relevant(row: dict) -> bool:
    """Return True if the entry matches any relevance term in title/abstract/keywords."""
    haystack = " ".join([
        _get(row, "title"),
        _get(row, "abstract"),
        _get(row, "author keywords"),
    ])
    return bool(_RELEVANCE_RE.search(haystack))


def should_exclude(row: dict) -> bool:
    """
    Return True if the entry should be excluded based on document type,
    unless it has more than 20 citations.
    """
    doc_type = _get(row, "document type").strip().lower()
    if doc_type in EXCLUDED_DOC_TYPES:
        cited = _parse_cited_by(_get(row, "cited by"))
        if cited <= 20:
            return True
    return False


def row_to_bibtex(row: dict, key: str) -> str:
    """Convert a normalised Scopus CSV row to a BibTeX entry string."""
    doc_type = _get(row, "document type")
    entry_type = _map_document_type(doc_type)

    authors = _clean_author_bibtex(_get(row, "authors", "author full names"))
    title = _get(row, "title")
    year = _get(row, "year")
    journal = _get(row, "source title")
    volume = _get(row, "volume")
    issue = _get(row, "issue")
    page_start = _get(row, "page start")
    page_end = _get(row, "page end")
    art_no = _get(row, "art. no.")
    doi = _get(row, "doi")
    abstract = _get(row, "abstract")
    keywords = _get(row, "author keywords")

    pages = _build_pages(page_start, page_end, art_no)

    # Build fields list
    fields: list[tuple[str, str]] = []
    if authors:
        fields.append(("author", _escape_bibtex(authors)))
    if title:
        fields.append(("title", "{" + _escape_bibtex(title) + "}"))
    if journal:
        if entry_type == "inproceedings":
            fields.append(("booktitle", _escape_bibtex(journal)))
        else:
            fields.append(("journal", _escape_bibtex(journal)))
    if year:
        fields.append(("year", year))
    if volume:
        fields.append(("volume", volume))
    if issue:
        fields.append(("number", issue))
    if pages:
        fields.append(("pages", pages))
    if doi:
        fields.append(("doi", doi))
    if keywords:
        fields.append(("keywords", _escape_bibtex(keywords)))

    # Format
    lines = [f"@{entry_type}{{{key},"]
    for fname, fval in fields:
        lines.append(f"  {fname:<12}= {{{fval}}},")
    lines.append("}")
    return "\n".join(lines)


def extract_existing_dois(bib_path: Path) -> set[str]:
    """
    Parse DOIs from an existing .bib file (simple regex approach).
    Returns a set of lowercased DOI strings.
    """
    dois: set[str] = set()
    if not bib_path.exists():
        return dois
    text = bib_path.read_text(encoding="utf-8")
    for match in re.finditer(r"doi\s*=\s*\{([^}]+)\}", text, re.IGNORECASE):
        dois.add(match.group(1).strip().lower())
    return dois


def extract_existing_keys(bib_path: Path) -> set[str]:
    """
    Parse citation keys from an existing .bib file.
    Returns a set of citation key strings.
    """
    keys: set[str] = set()
    if not bib_path.exists():
        return keys
    text = bib_path.read_text(encoding="utf-8")
    for match in re.finditer(r"@\w+\{([^,]+),", text):
        keys.add(match.group(1).strip())
    return keys


def deduplicate_key(key: str, existing_keys: set[str]) -> str:
    """
    Ensure the citation key is unique. If it already exists, append a
    lowercase letter suffix (a, b, c, ...).
    """
    if key not in existing_keys:
        return key
    for suffix_ord in range(ord("a"), ord("z") + 1):
        candidate = f"{key}{chr(suffix_ord)}"
        if candidate not in existing_keys:
            return candidate
    # Extremely unlikely fallback
    return f"{key}_{id(key)}"


def main() -> None:
    # Collect existing DOIs and keys from references.bib
    existing_dois = extract_existing_dois(BIB_FILE)
    existing_keys = extract_existing_keys(BIB_FILE)

    # Track DOIs we add during this run to avoid cross-CSV duplicates
    added_dois: set[str] = set()
    added_keys: set[str] = set(existing_keys)

    new_entries: list[str] = []
    report: dict[str, dict[str, int]] = {}

    for label, csv_path in CSV_FILES.items():
        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        print(f"  File: {csv_path.name}")
        print(f"{'='*60}")

        rows = read_scopus_csv(csv_path)
        total = len(rows)

        # Step 1: relevance filter
        relevant_rows = [r for r in rows if is_relevant(r)]
        after_relevance = len(relevant_rows)

        # Step 2: exclude unwanted document types (unless highly cited)
        filtered_rows = [r for r in relevant_rows if not should_exclude(r)]
        after_type_filter = len(filtered_rows)

        # Step 3: deduplicate by DOI (against existing bib and across CSVs)
        deduped_rows: list[dict] = []
        dupes_skipped = 0
        for r in filtered_rows:
            doi = _get(r, "doi").strip().lower()
            if doi:
                if doi in existing_dois or doi in added_dois:
                    dupes_skipped += 1
                    continue
                added_dois.add(doi)
            # If no DOI, we cannot deduplicate by DOI -- include it
            deduped_rows.append(r)

        after_dedup = len(deduped_rows)

        # Step 4: convert to BibTeX
        count_added = 0
        for r in deduped_rows:
            authors_raw = _get(r, "authors", "author full names")
            year = _get(r, "year")
            title = _get(r, "title")

            raw_key = _make_citation_key(authors_raw, year, title)
            key = deduplicate_key(raw_key, added_keys)
            added_keys.add(key)

            entry = row_to_bibtex(r, key)
            new_entries.append(entry)
            count_added += 1

        report[label] = {
            "total": total,
            "after_relevance": after_relevance,
            "after_type_filter": after_type_filter,
            "dupes_skipped": dupes_skipped,
            "after_dedup": after_dedup,
            "added": count_added,
        }

        print(f"  Total entries in CSV:        {total}")
        print(f"  After relevance filter:      {after_relevance}")
        print(f"  After doc-type filter:       {after_type_filter}")
        print(f"  Duplicates (DOI) skipped:    {dupes_skipped}")
        print(f"  After deduplication:         {after_dedup}")
        print(f"  Entries to add:              {count_added}")

    # Append to references.bib
    if new_entries:
        with open(BIB_FILE, "a", encoding="utf-8") as fh:
            fh.write("\n")
            fh.write("\n\n".join(new_entries))
            fh.write("\n")
        print(f"\nAppended {len(new_entries)} new entries to {BIB_FILE}")
    else:
        print("\nNo new entries to append.")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    grand_total = 0
    grand_relevant = 0
    grand_type = 0
    grand_dedup = 0
    grand_added = 0
    for label, stats in report.items():
        print(f"\n  {label}:")
        print(f"    Total in CSV:            {stats['total']}")
        print(f"    After relevance filter:  {stats['after_relevance']}")
        print(f"    After doc-type filter:   {stats['after_type_filter']}")
        print(f"    Duplicates skipped:      {stats['dupes_skipped']}")
        print(f"    After deduplication:     {stats['after_dedup']}")
        print(f"    Entries added:           {stats['added']}")
        grand_total += stats["total"]
        grand_relevant += stats["after_relevance"]
        grand_type += stats["after_type_filter"]
        grand_dedup += stats["after_dedup"]
        grand_added += stats["added"]

    print(f"\n  GRAND TOTALS:")
    print(f"    Total entries across CSVs:   {grand_total}")
    print(f"    After relevance filter:      {grand_relevant}")
    print(f"    After doc-type filter:       {grand_type}")
    print(f"    After deduplication:         {grand_dedup}")
    print(f"    Final entries appended:      {grand_added}")
    print(f"\n  Existing entries in bib:       {len(existing_keys)}")
    print(f"  New total entries in bib:      {len(existing_keys) + grand_added}")
    print("=" * 60)


if __name__ == "__main__":
    main()
