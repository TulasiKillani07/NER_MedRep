# synonym_mapper.py
# Step 3 of the preprocessing pipeline
# Maps informal / colloquial symptom terms to their canonical medical equivalents
# Example: "tummy ache" -> "stomach pain", "high temp" -> "fever"
# This ensures the NER model sees terms it was trained on

import json
from pathlib import Path

SYNONYMS_PATH = Path(__file__).parent.parent / "data" / "patterns" / "synonyms.json"


def _load_synonyms() -> dict:
    """Load synonym mapping from JSON file."""
    if SYNONYMS_PATH.exists():
        with open(SYNONYMS_PATH, "r") as f:
            return json.load(f)
    return {}


# Load once at module level
_synonyms = _load_synonyms()


def map_synonyms(text: str) -> str:
    """
    Replace informal/colloquial terms with canonical medical terms.
    Matches are case-insensitive and whole-phrase based.
    Longer phrases are matched first to avoid partial replacements.
    """
    # Sort by length descending so longer phrases match before shorter ones
    # e.g., "shortness of breath" matches before "breath"
    sorted_synonyms = sorted(_synonyms.items(), key=lambda x: len(x[0]), reverse=True)

    text_lower = text.lower()

    for informal, canonical in sorted_synonyms:
        if informal in text_lower:
            text_lower = text_lower.replace(informal, canonical)

    return text_lower
