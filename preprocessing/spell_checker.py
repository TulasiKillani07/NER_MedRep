# spell_checker.py
# Step 2 of the preprocessing pipeline
# Corrects spelling mistakes in user input
# Uses symspellpy for fast CPU-based spell correction
# Protects medical terms using a whitelist so they are never altered

import os
from pathlib import Path
from symspellpy import SymSpell, Verbosity

# Path to the medical whitelist file
WHITELIST_PATH = Path(__file__).parent.parent / "data" / "patterns" / "medical_whitelist.txt"

# symspellpy needs a frequency dictionary to work
# We use the built-in English frequency dictionary that ships with symspellpy
DICT_PATH = Path(__file__).parent.parent / "data" / "symspell_dict" / "frequency_dictionary_en_82_765.txt"
BIGRAM_PATH = Path(__file__).parent.parent / "data" / "symspell_dict" / "frequency_bigramdictionary_en_243_342.txt"

# Max edit distance for corrections (2 is standard — catches most typos)
MAX_EDIT_DISTANCE = 2


def _load_whitelist() -> set:
    """Load medical terms that should never be spell-corrected."""
    whitelist = set()
    if WHITELIST_PATH.exists():
        with open(WHITELIST_PATH, "r") as f:
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    whitelist.add(line)
    return whitelist


def _load_symspell() -> SymSpell:
    """Initialize and return a SymSpell instance with dictionary loaded."""
    sym_spell = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE, prefix_length=7)

    if DICT_PATH.exists():
        sym_spell.load_dictionary(str(DICT_PATH), term_index=0, count_index=1)
    else:
        # Fallback: use symspellpy's built-in corpus
        import pkg_resources
        dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
        sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    return sym_spell


# Load once at module level so it's not reloaded on every request
_sym_spell = _load_symspell()
_whitelist = _load_whitelist()


def correct(text: str) -> str:
    """
    Correct spelling in the input text.
    Words in the medical whitelist are never modified.
    Numbers and short words are preserved.
    Returns corrected text as a string.
    """
    words = text.split()
    corrected_words = []

    for word in words:
        # Preserve numbers (including those with letters like "3days")
        if any(char.isdigit() for char in word):
            corrected_words.append(word)
            continue
        
        # Never correct whitelisted medical terms
        if word.lower() in _whitelist:
            corrected_words.append(word)
            continue
        
        # Don't correct very short words (1-2 chars) unless they're clearly wrong
        if len(word) <= 2:
            corrected_words.append(word)
            continue

        suggestions = _sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=MAX_EDIT_DISTANCE)

        if suggestions:
            suggestion = suggestions[0].term
            # Only apply correction if edit distance is 1 (single typo)
            # This prevents aggressive corrections like "fver" -> "over"
            if suggestions[0].distance == 1:
                corrected_words.append(suggestion)
            else:
                # For distance > 1, keep original to avoid over-correction
                corrected_words.append(word)
        else:
            # No suggestion found, keep original
            corrected_words.append(word)

    return " ".join(corrected_words)
