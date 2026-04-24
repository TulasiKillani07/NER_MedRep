# normalizer.py
# Step 1 of the preprocessing pipeline
# Cleans raw user input before any NLP processing

import re


# Common medical abbreviations to expand before processing
ABBREVIATIONS = {
    "bp": "blood pressure",
    "hb": "hemoglobin",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "temp": "temperature",
    "wt": "weight",
    "ht": "height",
    "bmi": "body mass index",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "ha": "headache",
    "n/v": "nausea vomiting",
    "n/a": "",
    "abd": "abdominal",
    "bilat": "bilateral",
    "c/o": "complains of",
    "h/o": "history of",
    "k/c/o": "known case of",
}


def normalize(text: str) -> str:
    """
    Full normalization pipeline for a raw user query.
    Returns a cleaned, lowercase string ready for synonym mapping and NER.
    """
    text = _lowercase(text)
    text = _remove_extra_spaces(text)
    text = _remove_special_characters(text)
    text = _expand_abbreviations(text)
    text = _remove_extra_spaces(text)  # run again after expansion
    return text.strip()


def _lowercase(text: str) -> str:
    return text.lower()


def _remove_extra_spaces(text: str) -> str:
    # Replace multiple spaces, tabs, newlines with a single space
    return re.sub(r"\s+", " ", text)


def _remove_special_characters(text: str) -> str:
    # Keep letters, numbers, spaces, commas, periods, apostrophes, hyphens
    # Remove everything else (@, #, $, !, *, etc.)
    return re.sub(r"[^a-z0-9\s,.\'\-/]", "", text)


def _expand_abbreviations(text: str) -> str:
    # Replace known abbreviations with full forms
    # Uses word boundary matching to avoid partial replacements
    for abbr, full in ABBREVIATIONS.items():
        # escape special chars in abbreviation for regex
        escaped = re.escape(abbr)
        text = re.sub(rf"\b{escaped}\b", full, text)
    return text
