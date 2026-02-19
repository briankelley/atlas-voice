"""Atlas Voice text processing — punctuation and word replacement pipeline."""

import re
from logging_utils import log_debug


def strip_wake_phrase(text, config):
    """Remove the wake phrase from the beginning of transcribed text."""
    wake_phrase = config.get('wake_phrase', '')
    if not wake_phrase:
        return text
    pattern = r'^' + re.escape(wake_phrase) + r'[,.\s]*'
    result = re.sub(pattern, '', text, count=1, flags=re.IGNORECASE)
    if result != text:
        log_debug(f"[TEXT] Stripped wake phrase '{wake_phrase}' from transcription")
    return result.strip()


def process_text(text, config):
    """Apply punctuation conversion and word replacements from config."""
    # Apply spoken punctuation rules (from config, pre-built as (pattern_str, replacement) tuples)
    for pattern, replacement in config['spoken_punctuation']:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Apply word replacements (from config)
    for wrong, right in config['word_replacements'].items():
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text, flags=re.IGNORECASE)

    # Clean up whitespace
    text = re.sub(r' +', ' ', text)                    # Multiple spaces -> single
    text = re.sub(r' ([.,!?;:\)\]\}])', r'\1', text)   # Remove space before closing punct
    text = re.sub(r'([(\[\{]) ', r'\1', text)          # Remove space after opening punct

    return text.strip()


def contains_break_keyword(text, config):
    """Check if text contains the session end phrase (fuzzy match for 'break')."""
    # Fuzzy match for "break" catches Whisper mishearings: brick, brake, etc.
    end_session_pattern = r'\bbr[ei][ae]k\b'
    return bool(re.search(end_session_pattern, text, re.IGNORECASE))


def remove_break_keyword(text, config):
    """Remove the break keyword from text, return cleaned text."""
    end_session_pattern = r'\bbr[ei][ae]k\b[,.\s]*'
    return re.sub(end_session_pattern, '', text, flags=re.IGNORECASE).strip()
