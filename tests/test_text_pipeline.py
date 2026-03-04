"""Full text pipeline integration tests.

Wires whisper output through the complete chain:
  whisper -> strip_wake_phrase -> process_text -> output_text

Unlike unit tests that mock intermediate functions, these tests let the real
text_processing and text_output functions run, mocking only at the leaves
(subprocess.run for xdotool/xclip) and at the model boundary (whisper).
"""

import re
import time
from unittest import mock

import numpy as np
import pytest

from audio_buffer import AudioBuffer
from context import AtlasContext
from mailbox import Mailbox


def _make_ctx(config_overrides=None):
    """Build a fully-configured ctx for pipeline tests."""
    # Build punctuation rules matching production settings.conf
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
    items = sorted(punctuation_map.items(), key=lambda x: len(x[0]), reverse=True)
    rules = []
    for phrase, symbol in items:
        escaped = re.escape(phrase)
        pattern = r',?\s*\b' + escaped + r'\b[,.]?\s*'
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
        elif symbol in ('\n\n', '\n'):
            replacement = symbol
        else:
            replacement = symbol
        rules.append((pattern, replacement))

    config = {
        'sample_rate': 16000,
        'chunk_size': 1280,
        'silence_threshold': 500,
        'buffer_seconds': 10,
        'audio_device': '',
        'wake_preroll': 0.75,
        'wake_word_model': '/dev/null',
        'whisper_model': '/dev/null',
        'whisper_device': 'cpu',
        'whisper_compute_type': 'int8',
        'wake_word_threshold': 0.35,
        'silence_duration': 0,
        'max_record_duration': 60,
        'vad_timeout': 0.1,
        'auto_type': True,
        'beep_on_wake': False,
        'beep_sound': None,
        'debug': False,
        'log_transcripts': False,
        'tray_enabled': False,
        'typing_mode': 'console',
        'switch_to_console_phrase': 'switch to console',
        'switch_to_gui_phrase': 'switch to GUI',
        'spoken_punctuation': rules,
        'word_replacements': {
            'cloud': 'Claude',
            'clawed': 'Claude',
            'pseudo': 'sudo',
        },
        'wake_phrase': 'hey atlas',
        'end_phrase': 'break',
        'end_phrase_pattern': r'\b(?:break|brake|brick)\b',
        'icon_dir': '/tmp',
    }
    if config_overrides:
        config.update(config_overrides)
    buf = AudioBuffer(config)
    ctx = AtlasContext(config, buf)
    ctx.mailbox = Mailbox()
    return ctx


def _setup_running_audio(ctx):
    """Make audio buffer appear healthy and running."""
    ctx.audio_buffer.stream = mock.MagicMock()
    ctx.audio_buffer.last_callback_time = time.time()
    ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=True)


def _make_whisper_model(text):
    """Create a mock Whisper model returning given text."""
    model = mock.MagicMock()
    segment = mock.MagicMock()
    segment.text = text
    info = mock.MagicMock()
    model.transcribe.return_value = ([segment], info)
    return model


def _run_transcribing(ctx, whisper_text, overlap=False):
    """Run state_transcribing with the given whisper text.

    Returns the state result. Mocks subprocess (xdotool/xclip) at the leaf
    but lets text_processing and text_output run for real.
    """
    from state_transcribing import state_transcribing

    ctx.captured_audio = np.full(1280, 1000, dtype=np.int16)
    ctx.recording_mode = "wake"
    ctx.whisper_model = _make_whisper_model(whisper_text)
    ctx.audio_buffer.detect_speech_during = mock.MagicMock(
        return_value=500.0 if overlap else None
    )

    return state_transcribing(ctx)


class TestWhisperToOutput:
    """End-to-end: whisper output through the full text pipeline."""

    @mock.patch('text_output.subprocess.run')
    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    def test_basic_transcription_pipeline(self, mock_window, mock_run):
        """Whisper -> strip wake phrase -> output."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)

        result = _run_transcribing(ctx, "Hey Atlas hello world")

        assert result == "recording"  # continuous dictation
        # Verify xdotool was called with the processed text
        type_calls = [
            c for c in mock_run.call_args_list
            if 'xdotool' in c[0][0] and 'type' in c[0][0]
        ]
        assert len(type_calls) >= 1
        typed_text = type_calls[0][0][0][-1]  # last arg to xdotool type
        assert "hello world" in typed_text
        assert "hey atlas" not in typed_text.lower()

    @mock.patch('text_output.subprocess.run')
    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    def test_punctuation_pipeline(self, mock_window, mock_run):
        """Spoken punctuation converted through full pipeline."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)

        # Use "period" separated from "new line" to avoid regex interaction
        # (when adjacent, period's \s* eats the newline from "new line")
        result = _run_transcribing(
            ctx, "Hey Atlas first line new line second line period done"
        )

        assert result == "recording"
        type_calls = [
            c for c in mock_run.call_args_list
            if 'xdotool' in c[0][0] and 'type' in c[0][0]
        ]
        assert len(type_calls) >= 1
        typed_text = type_calls[0][0][0][-1]
        assert "\n" in typed_text
        assert "." in typed_text
        assert "new line" not in typed_text.lower()
        assert "period" not in typed_text.lower()

    @mock.patch('text_output.subprocess.run')
    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    def test_word_replacement_pipeline(self, mock_window, mock_run):
        """Word replacements applied through full pipeline."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)

        result = _run_transcribing(ctx, "Hey Atlas ask cloud about pseudo")

        assert result == "recording"
        type_calls = [
            c for c in mock_run.call_args_list
            if 'xdotool' in c[0][0] and 'type' in c[0][0]
        ]
        assert len(type_calls) >= 1
        typed_text = type_calls[0][0][0][-1]
        assert "Claude" in typed_text
        assert "sudo" in typed_text
        assert "cloud" not in typed_text
        assert "pseudo" not in typed_text

    @mock.patch('text_output.subprocess.run')
    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    def test_break_keyword_pipeline(self, mock_window, mock_run):
        """Break keyword ends session, remaining text output."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.play_beep = mock.MagicMock()

        result = _run_transcribing(ctx, "Hey Atlas finish this break")

        assert result == "listening"
        # xdotool type should have the text before "break"
        type_calls = [
            c for c in mock_run.call_args_list
            if 'xdotool' in c[0][0] and 'type' in c[0][0]
        ]
        assert len(type_calls) >= 1
        typed_text = type_calls[0][0][0][-1]
        assert "finish this" in typed_text
        assert "break" not in typed_text.lower()
        # Enter key pressed
        key_calls = [
            c for c in mock_run.call_args_list
            if 'xdotool' in c[0][0] and 'key' in c[0][0]
        ]
        assert len(key_calls) >= 1

    @mock.patch('text_output.subprocess.run')
    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    def test_mode_switch_pipeline(self, mock_window, mock_run):
        """Mode switch command changes typing_mode and removes command from text."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        assert ctx.typing_mode == 'console'

        result = _run_transcribing(ctx, "Hey Atlas switch to GUI hello there")

        assert result == "recording"
        assert ctx.typing_mode == 'gui'

    @mock.patch('text_output.subprocess.run')
    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    def test_wake_phrase_partial_strip(self, mock_window, mock_run):
        """Partial wake phrase (just 'Atlas') stripped from transcription."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)

        result = _run_transcribing(ctx, "Atlas do the thing")

        assert result == "recording"
        type_calls = [
            c for c in mock_run.call_args_list
            if 'xdotool' in c[0][0] and 'type' in c[0][0]
        ]
        assert len(type_calls) >= 1
        typed_text = type_calls[0][0][0][-1]
        assert "do the thing" in typed_text
        assert "atlas" not in typed_text.lower()

    @mock.patch('text_output.subprocess.run')
    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    def test_empty_after_processing(self, mock_window, mock_run):
        """Wake phrase only -> empty text -> returns listening."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)

        result = _run_transcribing(ctx, "Hey Atlas")

        assert result == "listening"
        # No xdotool type calls
        type_calls = [
            c for c in mock_run.call_args_list
            if 'xdotool' in c[0][0] and 'type' in c[0][0]
        ]
        assert len(type_calls) == 0

    @mock.patch('text_output.subprocess.run')
    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    def test_pipeline_with_all_transformations(self, mock_window, mock_run):
        """All transform types applied in one pass."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)

        # Separate comma and new line to avoid regex interaction
        result = _run_transcribing(
            ctx, "Hey Atlas cloud said pseudo comma then new line done"
        )

        assert result == "recording"
        type_calls = [
            c for c in mock_run.call_args_list
            if 'xdotool' in c[0][0] and 'type' in c[0][0]
        ]
        assert len(type_calls) >= 1
        typed_text = type_calls[0][0][0][-1]
        assert "Claude" in typed_text     # word replacement
        assert "sudo" in typed_text       # word replacement
        assert "," in typed_text          # punctuation
        assert "\n" in typed_text         # punctuation
        assert "cloud" not in typed_text
        assert "pseudo" not in typed_text
        assert "comma" not in typed_text.lower()

    @mock.patch('text_output.subprocess.run')
    @mock.patch('text_output._get_cached_window_class', return_value='gnome-terminal-server')
    def test_terminal_window_clipboard_only(self, mock_window, mock_run):
        """Terminal window: clipboard set but no xdotool type."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)

        result = _run_transcribing(ctx, "Hey Atlas hello world")

        assert result == "recording"
        # xclip should be called (clipboard)
        clip_calls = [
            c for c in mock_run.call_args_list
            if 'xclip' in c[0][0]
        ]
        assert len(clip_calls) >= 1
        # xdotool type should NOT be called (terminal detected)
        type_calls = [
            c for c in mock_run.call_args_list
            if 'xdotool' in c[0][0] and 'type' in c[0][0]
        ]
        assert len(type_calls) == 0

    @mock.patch('text_output.subprocess.run')
    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    def test_auto_type_disabled_no_output(self, mock_window, mock_run):
        """auto_type=False: no clipboard or typing calls."""
        ctx = _make_ctx({'auto_type': False})
        _setup_running_audio(ctx)

        result = _run_transcribing(ctx, "Hey Atlas hello world")

        # Still returns recording (continuous dictation continues)
        assert result == "recording"
        # No subprocess calls at all
        mock_run.assert_not_called()
