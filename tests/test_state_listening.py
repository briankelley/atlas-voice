"""Tests for state_listening.py - wake word detection loop and audio recovery."""

import time
import pytest
from unittest import mock

import numpy as np

from audio_buffer import AudioBuffer
from context import AtlasContext
from mailbox import Mailbox
from state_listening import state_listening


def _make_ctx(config_overrides=None):
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


def _setup_running_audio(ctx):
    """Make audio buffer appear running with a mock stream."""
    ctx.audio_buffer.stream = mock.MagicMock()
    ctx.audio_buffer.last_callback_time = time.time()


def _setup_wake_model(ctx, trigger=False, score=0.0):
    """Set up a mock wake model. If trigger=True, score exceeds threshold."""
    wake_model = mock.MagicMock()
    if trigger:
        score = 0.95
    wake_model.prediction_buffer = {
        'hey_atlas': [0.0, 0.0, score]
    }
    ctx.wake_model = wake_model
    return wake_model


def _mock_get_chunk(ctx, chunks):
    """Mock get_chunk to return chunks from a list, then None.

    state_listening flushes the chunk_queue on entry, so we can't pre-load
    the real queue. Instead, mock get_chunk to deliver chunks directly.
    Each entry is a (timestamp, np.array) tuple; get_chunk stores ts in
    last_dequeued_ts and returns the array.
    """
    chunk_iter = iter(chunks)

    def side_effect(timeout=0.08):
        try:
            ts, chunk = next(chunk_iter)
            ctx.audio_buffer.last_dequeued_ts = ts
            return chunk
        except StopIteration:
            return None

    ctx.audio_buffer.get_chunk = mock.MagicMock(side_effect=side_effect)


class TestListeningWakeDetection:
    """Wake word detection triggers transition to recording."""

    def test_wake_detected_returns_recording(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        _setup_wake_model(ctx, trigger=True)

        chunk = np.zeros(1280, dtype=np.int16)
        _mock_get_chunk(ctx, [(1000.0, chunk)])

        result = state_listening(ctx)

        assert result == "recording"
        assert ctx.recording_mode == "wake"
        assert ctx.wake_time is not None

    def test_wake_time_calculation(self):
        """wake_time = last_dequeued_ts - wake_preroll."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        _setup_wake_model(ctx, trigger=True)

        ts = 5000.0
        chunk = np.zeros(1280, dtype=np.int16)
        _mock_get_chunk(ctx, [(ts, chunk)])

        state_listening(ctx)

        expected = ts - ctx.config['wake_preroll']
        assert ctx.wake_time == pytest.approx(expected, abs=0.01)

    def test_wake_model_receives_int16(self):
        """Wake model predict() is called with int16 data."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        wake_model = _setup_wake_model(ctx, trigger=True)

        chunk = np.random.randint(-32768, 32767, size=1280, dtype=np.int16)
        _mock_get_chunk(ctx, [(1000.0, chunk)])

        state_listening(ctx)

        wake_model.predict.assert_called_once()
        call_arg = wake_model.predict.call_args[0][0]
        assert call_arg.dtype == np.int16

    def test_no_wake_processes_mailbox(self):
        """Without wake detection, mailbox is checked and QUIT exits."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        _setup_wake_model(ctx, trigger=False, score=0.01)

        chunk = np.zeros(1280, dtype=np.int16)
        _mock_get_chunk(ctx, [(1000.0, chunk)])

        # QUIT will be picked up on the mailbox check after the chunk is processed
        ctx.mailbox.post(Mailbox.QUIT)

        result = state_listening(ctx)
        assert result is None


class TestListeningMailboxInterrupts:
    """Mailbox messages interrupt the listening loop."""

    def test_toggle_pause_returns_paused(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        _setup_wake_model(ctx, trigger=False)

        chunk = np.zeros(1280, dtype=np.int16)
        _mock_get_chunk(ctx, [(1000.0, chunk)])
        ctx.mailbox.post(Mailbox.TOGGLE_PAUSE)

        result = state_listening(ctx)
        assert result == "paused"

    @mock.patch('context.gc')
    def test_toggle_enable_returns_disabled(self, mock_gc):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        _setup_wake_model(ctx, trigger=False)
        ctx.unload_models = mock.MagicMock()
        ctx.audio_buffer.stop = mock.MagicMock()

        chunk = np.zeros(1280, dtype=np.int16)
        _mock_get_chunk(ctx, [(1000.0, chunk)])
        ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)

        result = state_listening(ctx)

        assert result == "disabled"
        ctx.unload_models.assert_called_once()

    @mock.patch('context.gc')
    def test_quit_returns_none(self, mock_gc):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        _setup_wake_model(ctx, trigger=False)

        chunk = np.zeros(1280, dtype=np.int16)
        _mock_get_chunk(ctx, [(1000.0, chunk)])
        ctx.mailbox.post(Mailbox.QUIT)

        result = state_listening(ctx)
        assert result is None


class TestListeningEntryCleanup:
    """On entry, listening clears interstate data and flushes queue."""

    def test_clears_interstate_data(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        _setup_wake_model(ctx, trigger=True)

        # Set stale interstate data
        ctx.wake_time = 9999.0
        ctx.captured_audio = np.zeros(100, dtype=np.int16)
        ctx.recording_mode = "vad"

        chunk = np.zeros(1280, dtype=np.int16)
        _mock_get_chunk(ctx, [(1000.0, chunk)])

        state_listening(ctx)

        # After entry, recording_mode should be set to "wake" (new detection),
        # not the stale "vad"
        assert ctx.recording_mode == "wake"
        assert ctx.captured_audio is None  # cleared on entry

    def test_resets_wake_model(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        wake_model = _setup_wake_model(ctx, trigger=True)

        chunk = np.zeros(1280, dtype=np.int16)
        _mock_get_chunk(ctx, [(1000.0, chunk)])

        state_listening(ctx)

        wake_model.reset.assert_called_once()

    def test_flushes_chunk_queue(self):
        """Entry to listening flushes the chunk queue."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        _setup_wake_model(ctx, trigger=False)

        # Track that flush was called
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        _mock_get_chunk(ctx, [])  # no chunks
        ctx.mailbox.post(Mailbox.QUIT)

        state_listening(ctx)

        ctx.audio_buffer.flush_chunk_queue.assert_called()


class TestListeningAudioRecovery:
    """Audio health checks and device recovery."""

    @mock.patch('state_listening.time')
    @mock.patch('context.gc')
    def test_unhealthy_restart_succeeds_continues(self, mock_gc, mock_time):
        """Unhealthy stream that restarts successfully continues listening."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        _setup_wake_model(ctx, trigger=False)

        # time.time() calls in order:
        #   1. last_health_check = time.time()  -> 100.0
        #   2. now = time.time() in loop        -> 103.0  (triggers health check)
        #   3. now = time.time() next iteration  -> 103.0  (mailbox gets QUIT)
        mock_time.time.side_effect = [100.0, 103.0, 103.0, 103.0, 103.0]
        mock_time.monotonic = time.monotonic

        ctx.audio_buffer.is_healthy = mock.MagicMock(side_effect=[False])
        ctx.audio_buffer.restart = mock.MagicMock(return_value=True)
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        _mock_get_chunk(ctx, [])

        # Post QUIT after restart via side_effect on restart
        original_restart = ctx.audio_buffer.restart.side_effect

        def restart_then_quit():
            ctx.mailbox.post(Mailbox.QUIT)
            return True

        ctx.audio_buffer.restart = mock.MagicMock(side_effect=restart_then_quit)

        result = state_listening(ctx)

        ctx.audio_buffer.restart.assert_called_once()
        assert result is None

    @mock.patch('context.gc')
    def test_unhealthy_no_device_name_goes_disabled(self, mock_gc):
        """Unhealthy + restart fails + no device name -> disabled."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        _setup_wake_model(ctx, trigger=False)
        ctx.unload_models = mock.MagicMock()

        ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=False)
        ctx.audio_buffer.restart = mock.MagicMock(return_value=False)
        ctx.audio_buffer.device_name = ''
        _mock_get_chunk(ctx, [])  # no chunks
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        result = state_listening(ctx)

        assert result == "disabled"
        ctx.unload_models.assert_called()


class TestListeningNoStream:
    """Entering listening with no audio stream triggers start attempt."""

    def test_no_stream_starts_audio(self):
        ctx = _make_ctx()
        ctx.audio_buffer.stream = None
        _setup_wake_model(ctx, trigger=True)

        def fake_start():
            ctx.audio_buffer.stream = mock.MagicMock()

        ctx.audio_buffer.start = mock.MagicMock(side_effect=fake_start)

        chunk = np.zeros(1280, dtype=np.int16)
        _mock_get_chunk(ctx, [(1000.0, chunk)])

        result = state_listening(ctx)

        ctx.audio_buffer.start.assert_called_once()
        assert result == "recording"

    @mock.patch('context.gc')
    def test_no_stream_start_fails_no_device_goes_disabled(self, mock_gc):
        """No stream + start fails + no device name -> disabled."""
        ctx = _make_ctx()
        ctx.audio_buffer.stream = None
        ctx.audio_buffer.device_name = ''
        _setup_wake_model(ctx, trigger=False)

        ctx.audio_buffer.start = mock.MagicMock(side_effect=ValueError("No device"))

        result = state_listening(ctx)
        assert result == "disabled"
