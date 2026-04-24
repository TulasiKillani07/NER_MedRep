# ner_pipeline.py
# The main NER inference pipeline
# Loads the trained model + EntityRuler fallback
# Runs the full preprocessing pipeline before NER
# Returns structured entity results with negation flags

import json
from pathlib import Path

import spacy
from spacy.pipeline import EntityRuler
from loguru import logger

from preprocessing.normalizer import normalize
from preprocessing.spell_checker import correct
from preprocessing.synonym_mapper import map_synonyms
from preprocessing.negation_detector import apply_negation_to_entities

# Paths
MODEL_PATH = Path(__file__).parent / "output"
PATTERNS_DIR = Path(__file__).parent.parent / "data" / "patterns"

# Pattern files for EntityRuler fallback
PATTERN_FILES = [
    PATTERNS_DIR / "symptoms.json",
    PATTERNS_DIR / "durations.json",
    PATTERNS_DIR / "severities.json",
    PATTERNS_DIR / "indications.json",
]


def _load_patterns() -> list:
    """Load all EntityRuler patterns from JSON files."""
    all_patterns = []
    for path in PATTERN_FILES:
        if path.exists():
            with open(path, "r") as f:
                patterns = json.load(f)
                all_patterns.extend(patterns)
    return all_patterns


def load_pipeline():
    """
    Load the NER pipeline.
    - If trained model exists: load it and add EntityRuler as fallback
    - If no trained model yet: use base model + EntityRuler only
    """
    if MODEL_PATH.exists() and any(MODEL_PATH.iterdir()):
        logger.info("Loading trained model from disk")
        nlp = spacy.load(MODEL_PATH)
    else:
        logger.warning("No trained model found. Using base model + EntityRuler only.")
        nlp = spacy.load("en_core_web_md")

    # Add EntityRuler as fallback AFTER the NER component
    # overwrite_ents=False means EntityRuler only fills in what NER missed
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", after="ner", config={"overwrite_ents": False})
        patterns = _load_patterns()
        ruler.add_patterns(patterns)
        logger.info(f"EntityRuler loaded with {len(patterns)} patterns")

    return nlp


# Load pipeline once at module level (not on every request)
_nlp = load_pipeline()


def predict(text: str) -> dict:
    """
    Full inference pipeline:
    1. Normalize text
    2. Spell check
    3. Synonym mapping
    4. Run NER (trained model + EntityRuler fallback)
    5. Apply negation detection
    6. Return structured result

    Args:
        text: raw user input string

    Returns:
        dict with original text, processed text, and list of entities
    """
    original_text = text

    # Step 1: Normalize
    text = normalize(text)

    # Step 2: Spell check
    text = correct(text)

    # Step 3: Synonym mapping
    text = map_synonyms(text)

    # Step 4: Run NER
    doc = _nlp(text)

    # Step 5: Apply negation detection
    entities = apply_negation_to_entities(doc)

    # Filter to only include our custom labels
    filtered_entities = [
        ent for ent in entities 
        if ent.get("label") in {"SYMPTOM", "INDICATION", "SEVERITY", "DURATION"}
    ]

    return {
        "original_text": original_text,
        "processed_text": text,
        "entities": filtered_entities,
    }
