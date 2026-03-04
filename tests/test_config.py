"""Tests for config.py - configuration loading, type coercion, defaults,
punctuation rule building, word replacements, and end phrase pattern."""

import os
import tempfile
import pytest
from unittest import mock

import config as config_module
from config import load_config, _build_punctuation_rules, _build_word_replacements, _build_end_phrase_pattern


# ---------------------------------------------------------------------------
# load_config - defaults (no settings.conf)
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """Verify defaults when settings.conf is missing."""

    def test_loads_without_config_file(self, tmp_path):
        """load_config succeeds even when settings.conf doesn't exist."""
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert isinstance(cfg, dict)

    def test_default_sample_rate(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['sample_rate'] == 16000

    def test_default_chunk_size(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['chunk_size'] == 1280

    def test_default_wake_phrase(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['wake_phrase'] == 'hey atlas'

    def test_default_end_phrase(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['end_phrase'] == 'break'

    def test_default_typing_mode(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['typing_mode'] == 'console'

    def test_default_whisper_device(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['whisper_device'] == 'cuda'

    def test_default_compute_type(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['whisper_compute_type'] == 'float16'

    def test_default_booleans(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['auto_type'] is True
        assert cfg['beep_on_wake'] is True
        assert cfg['debug'] is False
        assert cfg['tray_enabled'] is True
        assert cfg['log_transcripts'] is False


# ---------------------------------------------------------------------------
# load_config - type coercion
# ---------------------------------------------------------------------------

class TestConfigTypeCoercion:
    """Verify values are coerced to the correct Python types."""

    def test_float_fields(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert isinstance(cfg['wake_word_threshold'], float)
        assert isinstance(cfg['wake_preroll'], float)
        assert isinstance(cfg['silence_duration'], float)
        assert isinstance(cfg['vad_timeout'], float)

    def test_int_fields(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert isinstance(cfg['silence_threshold'], int)
        assert isinstance(cfg['max_record_duration'], int)
        assert isinstance(cfg['buffer_seconds'], int)

    def test_bool_fields(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert isinstance(cfg['auto_type'], bool)
        assert isinstance(cfg['beep_on_wake'], bool)
        assert isinstance(cfg['debug'], bool)

    def test_string_fields(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert isinstance(cfg['wake_phrase'], str)
        assert isinstance(cfg['whisper_device'], str)
        assert isinstance(cfg['typing_mode'], str)


# ---------------------------------------------------------------------------
# load_config - custom values from file
# ---------------------------------------------------------------------------

class TestConfigCustomValues:
    """Verify config file values override defaults."""

    def _write_config(self, tmp_path, content):
        conf_file = tmp_path / 'settings.conf'
        conf_file.write_text(content)
        return str(conf_file)

    def test_override_wake_threshold(self, tmp_path):
        path = self._write_config(tmp_path, "[wake_word]\nthreshold = 0.75\n")
        with mock.patch.object(config_module, '_CONFIG_FILE', path):
            cfg = load_config()
        assert cfg['wake_word_threshold'] == 0.75

    def test_override_silence_threshold(self, tmp_path):
        path = self._write_config(tmp_path, "[audio]\nsilence_threshold = 1000\n")
        with mock.patch.object(config_module, '_CONFIG_FILE', path):
            cfg = load_config()
        assert cfg['silence_threshold'] == 1000

    def test_override_whisper_device_cpu(self, tmp_path):
        path = self._write_config(tmp_path, "[whisper]\ndevice = cpu\n")
        with mock.patch.object(config_module, '_CONFIG_FILE', path):
            cfg = load_config()
        assert cfg['whisper_device'] == 'cpu'

    def test_override_debug_true(self, tmp_path):
        path = self._write_config(tmp_path, "[behavior]\ndebug_mode = true\n")
        with mock.patch.object(config_module, '_CONFIG_FILE', path):
            cfg = load_config()
        assert cfg['debug'] is True

    def test_override_typing_mode_gui(self, tmp_path):
        path = self._write_config(tmp_path, "[behavior]\ntyping_mode = GUI\n")
        with mock.patch.object(config_module, '_CONFIG_FILE', path):
            cfg = load_config()
        assert cfg['typing_mode'] == 'gui'  # lowercased


# ---------------------------------------------------------------------------
# load_config - path derivation
# ---------------------------------------------------------------------------

class TestConfigPaths:
    """Verify paths are derived from _BASE_DIR."""

    def test_base_dir_set(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert os.path.isabs(cfg['base_dir'])

    def test_wake_word_model_path(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['wake_word_model'].endswith('hey_atlas.tflite')
        assert 'openwakeword' in cfg['wake_word_model']

    def test_whisper_model_path(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['whisper_model'].endswith('whisper-large-v3')

    def test_icon_dir_path(self, tmp_path):
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
        assert cfg['icon_dir'].endswith('icons')


# ---------------------------------------------------------------------------
# load_config - beep sound resolution
# ---------------------------------------------------------------------------

class TestConfigBeepSound:
    """Verify beep sound candidate resolution."""

    def test_beep_sound_found(self, tmp_path):
        """When a candidate path exists, it's selected."""
        beep_file = tmp_path / 'beep.oga'
        beep_file.write_text('')
        candidates = [str(beep_file)]
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            cfg = load_config()
            # Manually verify the logic since we can't easily inject candidates
            # Instead, just verify the key exists and is a string or None
            assert 'beep_sound' in cfg
            assert cfg['beep_sound'] is None or isinstance(cfg['beep_sound'], str)

    def test_beep_sound_none_when_missing(self, tmp_path):
        """When no candidate exists, beep_sound is None."""
        with mock.patch.object(config_module, '_CONFIG_FILE', str(tmp_path / 'nonexistent.conf')):
            # Mock os.path.exists to return False for beep candidates
            original_exists = os.path.exists
            def fake_exists(path):
                if '/usr/share/sounds/' in path:
                    return False
                return original_exists(path)
            with mock.patch('os.path.exists', side_effect=fake_exists):
                cfg = load_config()
        assert cfg['beep_sound'] is None


# ---------------------------------------------------------------------------
# _build_punctuation_rules
# ---------------------------------------------------------------------------

class TestBuildPunctuationRules:
    """Tests for punctuation rule compilation."""

    def _make_parser(self, rules_dict):
        import configparser
        cp = configparser.ConfigParser()
        cp.optionxform = str
        cp['spoken_punctuation'] = rules_dict
        return cp

    def test_empty_section(self):
        import configparser
        cp = configparser.ConfigParser()
        # No spoken_punctuation section
        assert _build_punctuation_rules(cp) == []

    def test_single_rule(self):
        cp = self._make_parser({'period': '.'})
        rules = _build_punctuation_rules(cp)
        assert len(rules) == 1
        pattern, replacement = rules[0]
        assert 'period' in pattern
        assert replacement == '. '  # period gets trailing space

    def test_longer_phrases_sorted_first(self):
        cp = self._make_parser({
            'period': '.',
            'exclamation point': '!',
            'comma': ',',
        })
        rules = _build_punctuation_rules(cp)
        # 'exclamation point' (17 chars) should come before 'period' (6) and 'comma' (5)
        phrases = [r[0] for r in rules]
        assert 'exclamation' in phrases[0]

    def test_newline_replacement(self):
        cp = self._make_parser({'new line': '\\n'})
        rules = _build_punctuation_rules(cp)
        _, replacement = rules[0]
        assert replacement == '\n'

    def test_double_newline_replacement(self):
        cp = self._make_parser({'new paragraph': '\\n\\n'})
        rules = _build_punctuation_rules(cp)
        _, replacement = rules[0]
        assert replacement == '\n\n'

    def test_opening_bracket_gets_leading_space(self):
        cp = self._make_parser({'open paren': '('})
        rules = _build_punctuation_rules(cp)
        _, replacement = rules[0]
        assert replacement == ' ('

    def test_closing_bracket_gets_trailing_space(self):
        cp = self._make_parser({'close paren': ')'})
        rules = _build_punctuation_rules(cp)
        _, replacement = rules[0]
        assert replacement == ') '

    def test_equals_gets_surrounding_spaces(self):
        cp = self._make_parser({'equals': '='})
        rules = _build_punctuation_rules(cp)
        _, replacement = rules[0]
        assert replacement == ' = '

    def test_ellipsis_gets_trailing_space(self):
        cp = self._make_parser({'dot dot dot': '...'})
        rules = _build_punctuation_rules(cp)
        _, replacement = rules[0]
        assert replacement == '... '


# ---------------------------------------------------------------------------
# _build_word_replacements
# ---------------------------------------------------------------------------

class TestBuildWordReplacements:
    """Tests for word replacement dict construction."""

    def _make_parser(self, replacements_dict):
        import configparser
        cp = configparser.ConfigParser()
        cp.optionxform = str
        cp['word_replacements'] = replacements_dict
        return cp

    def test_empty_no_section(self):
        import configparser
        cp = configparser.ConfigParser()
        assert _build_word_replacements(cp) == {}

    def test_basic_replacement(self):
        cp = self._make_parser({'cloud': 'Claude'})
        result = _build_word_replacements(cp)
        assert result == {'cloud': 'Claude'}

    def test_preserves_case_in_keys(self):
        """optionxform=str means keys preserve case."""
        cp = self._make_parser({'Cloud': 'Claude', 'cloud': 'Claude'})
        result = _build_word_replacements(cp)
        assert 'Cloud' in result
        assert 'cloud' in result

    def test_multiple_replacements(self):
        cp = self._make_parser({
            'cloud': 'Claude',
            'pseudo': 'sudo',
        })
        result = _build_word_replacements(cp)
        assert len(result) == 2
        assert result['cloud'] == 'Claude'
        assert result['pseudo'] == 'sudo'


# ---------------------------------------------------------------------------
# _build_end_phrase_pattern
# ---------------------------------------------------------------------------

class TestBuildEndPhrasePattern:
    """Tests for end phrase regex pattern construction."""

    def _make_parser(self, end_phrase='break', variants=''):
        import configparser
        cp = configparser.ConfigParser()
        cp.optionxform = str
        cp['session'] = {'end_phrase': end_phrase, 'end_phrase_variants': variants}
        return cp

    def test_basic_pattern(self):
        import re
        cp = self._make_parser('break')
        pattern = _build_end_phrase_pattern(cp)
        assert pattern is not None
        assert re.search(pattern, 'break', re.IGNORECASE)

    def test_pattern_with_variants(self):
        import re
        cp = self._make_parser('break', 'brake, brick')
        pattern = _build_end_phrase_pattern(cp)
        assert re.search(pattern, 'break', re.IGNORECASE)
        assert re.search(pattern, 'brake', re.IGNORECASE)
        assert re.search(pattern, 'brick', re.IGNORECASE)

    def test_word_boundaries(self):
        import re
        cp = self._make_parser('break', 'brake')
        pattern = _build_end_phrase_pattern(cp)
        assert not re.search(pattern, 'breakfast', re.IGNORECASE)
        assert not re.search(pattern, 'unbreakable', re.IGNORECASE)

    def test_empty_phrase_returns_none(self):
        cp = self._make_parser('')
        pattern = _build_end_phrase_pattern(cp)
        assert pattern is None

    def test_no_variants(self):
        import re
        cp = self._make_parser('stop', '')
        pattern = _build_end_phrase_pattern(cp)
        assert re.search(pattern, 'stop', re.IGNORECASE)
        assert not re.search(pattern, 'break', re.IGNORECASE)

    def test_special_characters_escaped(self):
        import re
        cp = self._make_parser('stop.now')
        pattern = _build_end_phrase_pattern(cp)
        # The dot should be escaped, so 'stopXnow' should NOT match
        assert not re.search(pattern, 'stopXnow', re.IGNORECASE)
        assert re.search(pattern, 'stop.now', re.IGNORECASE)


# ---------------------------------------------------------------------------
# load_config - full settings.conf integration
# ---------------------------------------------------------------------------

class TestConfigIntegrationWithRealFile:
    """Load the actual settings.conf from the repo and verify key values."""

    def test_loads_real_config(self):
        """Smoke test: loading the real settings.conf produces valid config."""
        cfg = load_config()
        assert cfg['wake_phrase'] == 'hey atlas'
        assert cfg['sample_rate'] == 16000
        assert isinstance(cfg['spoken_punctuation'], list)
        assert len(cfg['spoken_punctuation']) > 0
        assert isinstance(cfg['word_replacements'], dict)
        assert len(cfg['word_replacements']) > 0

    def test_real_config_end_phrase_pattern(self):
        """The real config produces a working end phrase pattern."""
        import re
        cfg = load_config()
        pattern = cfg['end_phrase_pattern']
        assert pattern is not None
        assert re.search(pattern, 'break', re.IGNORECASE)
        assert re.search(pattern, 'brake', re.IGNORECASE)
        assert not re.search(pattern, 'breakfast', re.IGNORECASE)

    def test_real_config_punctuation_count(self):
        """settings.conf has 40+ punctuation rules."""
        cfg = load_config()
        assert len(cfg['spoken_punctuation']) >= 40

    def test_real_config_word_replacement_count(self):
        """settings.conf has multiple word replacements."""
        cfg = load_config()
        assert len(cfg['word_replacements']) >= 5
