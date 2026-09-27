import re
import unicodedata


def normalize_query(query: str) -> str:
    """
    Normalize superficial formatting noise without
    changing the meaning of the query.

    Operations:
    - Unicode normalization
    - Remove leading/trailing whitespace
    - Collapse repeated whitespace
    - Convert to lowercase
    - Collapse repeated question marks
    - Collapse repeated exclamation marks
    """

    if not query:
        return ""

    # Normalize Unicode representation.
    query = unicodedata.normalize("NFKC", query)

    # Remove leading/trailing whitespace.
    query = query.strip()

    # Collapse spaces, tabs, and newlines.
    query = re.sub(r"\s+", " ", query)

    # Normalize case.
    query = query.lower()

    # Collapse repeated question marks.
    query = re.sub(r"\?{2,}", "?", query)

    # Collapse repeated exclamation marks.
    query = re.sub(r"!{2,}", "!", query)

    return query
