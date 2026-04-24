# negation_detector.py
# Step 4 of the preprocessing pipeline
# Detects negation context around entities AFTER NER runs
# Example: "no fever" -> fever entity is marked negated=True
# Uses a simple window-based approach (no ML needed)
# negspacy is used as the spaCy component for negation

# This module provides:
# 1. A function to add negation detection to a spaCy pipeline
# 2. A function to check if a detected entity is negated

# Negation cue words — if any of these appear before an entity within
# a window of N tokens, the entity is considered negated
NEGATION_CUES = {
    "no",
    "not",
    "without",
    "denies",
    "deny",
    "denied",
    "absence",
    "absent",
    "never",
    "neither",
    "nor",
    "free of",
    "negative for",
    "ruled out",
    "no evidence of",
    "no history of",
    "no complaint of",
}

# How many tokens before the entity to look for negation cues
NEGATION_WINDOW = 5


def is_negated(entity_span, doc) -> bool:
    """
    Check if an entity span is negated based on surrounding tokens.
    Looks NEGATION_WINDOW tokens before the entity start for negation cues.

    Args:
        entity_span: a spaCy Span object (the detected entity)
        doc: the full spaCy Doc object

    Returns:
        True if the entity appears to be negated, False otherwise
    """
    start = entity_span.start
    # Look at tokens before the entity within the window
    window_start = max(0, start - NEGATION_WINDOW)
    preceding_tokens = [token.text.lower() for token in doc[window_start:start]]
    preceding_text = " ".join(preceding_tokens)

    for cue in NEGATION_CUES:
        if cue in preceding_text:
            return True

    return False


def apply_negation_to_entities(doc) -> list:
    """
    Given a spaCy doc with entities already detected,
    return a list of entity dicts with negation flags applied.

    Returns:
        List of dicts:
        [
            {
                "text": "fever",
                "label": "SYMPTOM",
                "start": 3,
                "end": 8,
                "negated": False
            },
            ...
        ]
    """
    results = []
    for ent in doc.ents:
        negated = is_negated(ent, doc)
        results.append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "negated": negated,
        })
    return results
