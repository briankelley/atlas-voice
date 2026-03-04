"""Tests for state_disabled.py - disabled state transitions."""

import threading
import time
import pytest
from unittest import mock

from audio_buffer import AudioBuffer
from context import AtlasContext
from mailbox import Mailbox
from state_disabled import state_disabled


def _make_ctx(config_overrides=None):
    """Build a test context with mocked models and audio."""
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
    if config_overrides:
        config.update(config_overrides)
    buf = AudioBuffer(config)
    ctx = AtlasContext(config, buf)
    ctx.mailbox = Mailbox()
    return ctx


class TestDisabledToListening:
    """TOGGLE_ENABLE triggers model load + audio start -> listening."""

    @mock.patch('context.gc')
    def test_enable_success(self, mock_gc):
        ctx = _make_ctx()
        ctx.load_models = mock.MagicMock()
        ctx.audio_buffer.start = mock.MagicMock()
        ctx.audio_buffer.is_device_present = mock.MagicMock(return_value=True)

        # Post TOGGLE_ENABLE before entering state
        ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)
        result = state_disabled(ctx)

        assert result == "listening"
        ctx.load_models.assert_called_once()
        ctx.audio_buffer.start.assert_called_once()

    @mock.patch('context.gc')
    def test_enable_model_load_fails_stays_disabled(self, mock_gc):
        """Model load failure keeps us in disabled (loops back)."""
        ctx = _make_ctx()
        call_count = [0]

        def failing_load():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("GPU out of memory")

        ctx.load_models = mock.MagicMock(side_effect=failing_load)
        ctx.audio_buffer.start = mock.MagicMock()

        # First TOGGLE_ENABLE fails, second one succeeds
        def post_sequence():
            time.sleep(0.05)
            ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)
            time.sleep(0.3)
            ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)

        t = threading.Thread(target=post_sequence)
        t.start()
        result = state_disabled(ctx)
        t.join()

        assert result == "listening"
        assert call_count[0] == 2

    @mock.patch('context.gc')
    def test_enable_audio_start_fails_unloads_models(self, mock_gc):
        """Audio start failure after model load unloads models, stays disabled."""
        ctx = _make_ctx()
        ctx.load_models = mock.MagicMock()
        ctx.unload_models = mock.MagicMock()
        audio_fail_count = [0]

        def failing_audio():
            audio_fail_count[0] += 1
            if audio_fail_count[0] == 1:
                raise RuntimeError("No audio device")

        ctx.audio_buffer.start = mock.MagicMock(side_effect=failing_audio)

        # First enable fails audio, second succeeds
        def post_sequence():
            time.sleep(0.05)
            ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)
            time.sleep(0.3)
            ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)

        t = threading.Thread(target=post_sequence)
        t.start()
        result = state_disabled(ctx)
        t.join()

        assert result == "listening"
        # unload_models called on first failure
        ctx.unload_models.assert_called()

    @mock.patch('context.gc')
    def test_enable_device_not_present_stays_disabled(self, mock_gc):
        """Device presence check fails, stays disabled."""
        ctx = _make_ctx()
        ctx.audio_buffer.device_name = "Missing Device"
        ctx.audio_buffer.is_device_present = mock.MagicMock(return_value=False)
        ctx.load_models = mock.MagicMock()

        # First enable fails device check, then QUIT
        def post_sequence():
            time.sleep(0.05)
            ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)
            time.sleep(0.3)
            ctx.mailbox.post(Mailbox.QUIT)

        t = threading.Thread(target=post_sequence)
        t.start()
        result = state_disabled(ctx)
        t.join()

        assert result is None  # QUIT
        ctx.load_models.assert_not_called()


class TestDisabledQuit:
    """QUIT in disabled state triggers shutdown."""

    @mock.patch('context.gc')
    def test_quit_returns_none(self, mock_gc):
        ctx = _make_ctx()
        ctx.mailbox.post(Mailbox.QUIT)
        result = state_disabled(ctx)
        assert result is None


class TestDisabledIgnoresTogglePause:
    """TOGGLE_PAUSE is nonsensical in disabled and should be ignored."""

    @mock.patch('context.gc')
    def test_toggle_pause_ignored(self, mock_gc):
        ctx = _make_ctx()

        def post_sequence():
            time.sleep(0.05)
            ctx.mailbox.post(Mailbox.TOGGLE_PAUSE)  # ignored
            time.sleep(0.1)
            ctx.mailbox.post(Mailbox.QUIT)  # exits

        t = threading.Thread(target=post_sequence)
        t.start()
        result = state_disabled(ctx)
        t.join()

        assert result is None  # exited via QUIT, not pause
