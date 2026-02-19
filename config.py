"""Atlas Voice configuration loader."""

import os
import configparser
import re

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_BASE_DIR, "settings.conf")


def load_config():
    """Load configuration from settings.conf, with sensible defaults. Returns a dict."""
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case (default lowercases keys)

    # Defaults
    defaults = {
        'wake_word': {'threshold': '0.50', 'wake_preroll': '0.75', 'wake_phrase': 'hey atlas'},
        'audio': {
            'silence_threshold': '500',
            'silence_duration': '2.0',
            'max_record_duration': '60',
            'buffer_seconds': '120',
            'vad_timeout': '5.0',
            'device': '',
        },
        'whisper': {'device': 'cuda', 'compute_type': 'float16'},
        'behavior': {
            'auto_type': 'true',
            'beep_on_wake': 'true',
            'debug_mode': 'false',
            'tray_enabled': 'true',
            'typing_mode': 'console',
            'switch_to_console_phrase': 'switch to console',
            'switch_to_gui_phrase': 'switch to gui',
        },
    }

    for section, values in defaults.items():
        config[section] = values

    if os.path.exists(_CONFIG_FILE):
        config.read(_CONFIG_FILE)

    # Build result dict with typed values
    result = {
        # Paths
        'base_dir': _BASE_DIR,
        'wake_word_model': os.path.join(_BASE_DIR, "models", "openwakeword", "hey_atlas.tflite"),
        'whisper_model': os.path.join(_BASE_DIR, "models", "whisper-large-v3"),
        'icon_dir': os.path.join(_BASE_DIR, "icons"),

        # Wake word
        'wake_word_threshold': config.getfloat('wake_word', 'threshold'),
        'wake_preroll': config.getfloat('wake_word', 'wake_preroll'),
        'wake_phrase': config.get('wake_word', 'wake_phrase').strip(),

        # Audio
        'sample_rate': 16000,
        'chunk_size': 1280,  # 80ms chunks (16000 * 0.08)
        'silence_threshold': config.getint('audio', 'silence_threshold'),
        'silence_duration': config.getfloat('audio', 'silence_duration'),
        'max_record_duration': config.getint('audio', 'max_record_duration'),
        'buffer_seconds': config.getint('audio', 'buffer_seconds'),
        'vad_timeout': config.getfloat('audio', 'vad_timeout'),
        'audio_device': config.get('audio', 'device').strip(),

        # Whisper
        'whisper_device': config.get('whisper', 'device'),
        'whisper_compute_type': config.get('whisper', 'compute_type'),

        # Behavior
        'auto_type': config.getboolean('behavior', 'auto_type'),
        'beep_on_wake': config.getboolean('behavior', 'beep_on_wake'),
        'debug': config.getboolean('behavior', 'debug_mode'),
        'tray_enabled': config.getboolean('behavior', 'tray_enabled'),
        'typing_mode': config.get('behavior', 'typing_mode').lower(),
        'switch_to_console_phrase': config.get('behavior', 'switch_to_console_phrase').strip(),
        'switch_to_gui_phrase': config.get('behavior', 'switch_to_gui_phrase').strip(),

        # Logging
        'log_transcripts': config.getboolean('behavior', 'log_transcripts') if config.has_option('behavior', 'log_transcripts') else False,

        # Beep sound
        'beep_sound': None,  # Will be set below
    }

    # Find beep sound
    beep_candidates = [
        "/usr/share/sounds/freedesktop/stereo/audio-volume-change.oga",
        "/usr/share/sounds/freedesktop/stereo/bell.oga",
    ]
    for path in beep_candidates:
        if os.path.exists(path):
            result['beep_sound'] = path
            break

    # Build punctuation rules from config
    result['spoken_punctuation'] = _build_punctuation_rules(config)

    # Build word replacements from config
    result['word_replacements'] = _build_word_replacements(config)

    return result


def _build_punctuation_rules(config):
    """Build regex patterns from config's spoken_punctuation section."""
    rules = []
    if 'spoken_punctuation' not in config:
        return rules

    # Sort by phrase length descending (longer phrases first)
    items = list(config['spoken_punctuation'].items())
    items.sort(key=lambda x: len(x[0]), reverse=True)

    for phrase, replacement in items:
        escaped = re.escape(phrase)
        pattern = r',?\s*\b' + escaped + r'\b[,.]?\s*'
        # Handle special replacements
        if replacement == '\\n\\n':
            replacement = '\n\n'
        elif replacement == '\\n':
            replacement = '\n'
        elif replacement == '\\\\':
            replacement = '\\\\'
        # Add spacing based on punctuation type
        if replacement in '.!?':
            replacement = replacement + ' '
        elif replacement in ',;:':
            replacement = replacement + ' '
        elif replacement in '([{':
            replacement = ' ' + replacement
        elif replacement in ')]}':
            replacement = replacement + ' '
        elif replacement in '=-+':
            replacement = ' ' + replacement + ' '
        elif replacement == '...':
            replacement = '... '
        rules.append((pattern, replacement))

    return rules


def _build_word_replacements(config):
    """Build word replacement dict from config."""
    replacements = {}
    if 'word_replacements' in config:
        for wrong, correct in config['word_replacements'].items():
            replacements[wrong] = correct
    return replacements
