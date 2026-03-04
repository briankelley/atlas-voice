"""Tests for state_paused.py - paused state transitions."""

import pytest
from unittest import mock

from audio_buffer import AudioBuffer
from context import AtlasContext
from mailbox import Mailbox
from state_paused import state_paused


def _make_ctx():
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
        'wake_word_threshold': 0.50,
        'silence_duration': 2.0,
        'max_record_duration': 60,
        'vad_timeout': 5.0,
        'auto_type': False,
        'beep_on_wake': False,
        'debug': False,
        'log_transcripts': False,
        'tray_enabled': False,
        'typing_mode': 'console',
        'switch_to_console_phrase': '',
        'switch_to_gui_phrase': '',
        'spoken_punctuation': [],
        'word_replacements': {},
        'icon_dir': '/tmp',
    }
    buf = AudioBuffer(config)
    ctx = AtlasContext(config, buf)
    ctx.mailbox = Mailbox()
    return ctx


class TestPausedToListening:
    """TOGGLE_PAUSE in paused -> listening."""

    def test_toggle_pause_returns_listening(self):
        ctx = _make_ctx()
        ctx.mailbox.post(Mailbox.TOGGLE_PAUSE)
        result = state_paused(ctx)
        assert result == "listening"


class TestPausedToDisabled:
    """TOGGLE_ENABLE in paused -> disabled (unloads models, stops audio)."""

    @mock.patch('context.gc')
    def test_toggle_enable_returns_disabled(self, mock_gc):
        ctx = _make_ctx()
        ctx.unload_models = mock.MagicMock()
        ctx.audio_buffer.stop = mock.MagicMock()
        ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)

        result = state_paused(ctx)

        assert result == "disabled"
        ctx.unload_models.assert_called_once()
        ctx.audio_buffer.stop.assert_called_once()


class TestPausedQuit:
    """QUIT in paused -> shutdown."""

    @mock.patch('context.gc')
    def test_quit_returns_none(self, mock_gc):
        ctx = _make_ctx()
        ctx.mailbox.post(Mailbox.QUIT)
        result = state_paused(ctx)
        assert result is None


class TestPausedSetsIcon:
    """Paused state sets icon to AV_OFF."""

    def test_icon_set_to_off(self):
        ctx = _make_ctx()
        ctx.set_icon = mock.MagicMock()
        ctx.mailbox.post(Mailbox.QUIT)

        state_paused(ctx)

        ctx.set_icon.assert_called_with("AV_OFF")
