"""Shared fixtures for Atlas Voice tests."""

import pytest


@pytest.fixture
def minimal_config():
    """Minimal config dict with defaults suitable for unit tests."""
    return {
        'sample_rate': 16000,
        'chunk_size': 1280,
        'silence_threshold': 500,
        'buffer_seconds': 10,
        'audio_device': '',
        'wake_preroll': 0.75,
        'wake_phrase': 'hey atlas',
        'end_phrase': 'break',
        'end_phrase_pattern': r'\b(?:break|brake|brick)\b',
        'spoken_punctuation': [],
        'word_replacements': {},
        'debug': False,
    }


@pytest.fixture
def punctuation_config(minimal_config):
    """Config with spoken punctuation rules matching settings.conf patterns."""
    import re
    rules = []
    punctuation_map = {
        'open parenthesis': '(',
        'close parenthesis': ')',
        'exclamation point': '!',
        'question mark': '?',
        'new paragraph': '\n\n',
        'new line': '\n',
        'dot dot dot': '...',
        'period': '.',
        'comma': ',',
        'colon': ':',
        'semicolon': ';',
        'dash': '-',
    }
    # Sort by phrase length descending (matches production behavior)
    items = sorted(punctuation_map.items(), key=lambda x: len(x[0]), reverse=True)
    for phrase, symbol in items:
        escaped = re.escape(phrase)
        pattern = r',?\s*\b' + escaped + r'\b[,.]?\s*'
        # Apply spacing rules matching _build_punctuation_rules
        if symbol in '.!?':
            replacement = symbol + ' '
        elif symbol in ',;:':
            replacement = symbol + ' '
        elif symbol in '([{':
            replacement = ' ' + symbol
        elif symbol in ')]}':
            replacement = symbol + ' '
        elif symbol in '=-+':
            replacement = ' ' + symbol + ' '
        elif symbol == '...':
            replacement = '... '
        elif symbol == '\n\n':
            replacement = '\n\n'
        elif symbol == '\n':
            replacement = '\n'
        else:
            replacement = symbol
        rules.append((pattern, replacement))
    minimal_config['spoken_punctuation'] = rules
    return minimal_config


@pytest.fixture
def word_replacement_config(minimal_config):
    """Config with word replacements matching settings.conf."""
    minimal_config['word_replacements'] = {
        'cloud': 'Claude',
        'clawed': 'Claude',
        'pseudo': 'sudo',
    }
    return minimal_config
