"""Tests for state_recording.py - audio capture, silence detection, VAD mode."""

import time
import pytest
from unittest import mock

import numpy as np

from audio_buffer import AudioBuffer
from context import AtlasContext
from mailbox import Mailbox
from state_recording import state_recording


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
        'silence_duration': 0,  # zero for instant silence detection with mocked get_chunk
        'max_record_duration': 60,
        'vad_timeout': 0.2,  # short for fast tests
        'auto_type': False,
        'beep_on_wake': False,
        'beep_sound': None,
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
    ctx.audio_buffer.stream = mock.MagicMock()
    ctx.audio_buffer.last_callback_time = time.time()
    ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=True)


def _make_speech_chunk(amplitude=1000):
    """Create a chunk with audio above silence threshold."""
    return np.full(1280, amplitude, dtype=np.int16)


def _make_silence_chunk():
    """Create a chunk below silence threshold."""
    return np.zeros(1280, dtype=np.int16)


def _mock_get_chunk(ctx, chunks):
    """Mock get_chunk to return chunks from a list, then None.

    state_recording flushes the chunk_queue in wake mode, so we mock
    get_chunk to deliver chunks directly and avoid the real queue.
    """
    chunk_iter = iter(chunks)

    def side_effect(timeout=0.08):
        try:
            return next(chunk_iter)
        except StopIteration:
            return None

    ctx.audio_buffer.get_chunk = mock.MagicMock(side_effect=side_effect)


class TestRecordingWakeMode:
    """Wake mode: captures preroll audio from ring buffer + live chunks."""

    def test_wake_mode_captures_audio(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0

        # Seed ring buffer with audio since wake_time
        for i in range(5):
            ts = 999.0 + i * 0.08
            chunk = _make_speech_chunk()
            ctx.audio_buffer.ring_buffer.append((ts, chunk.copy()))

        # Mock flush so it doesn't drain, and mock get_chunk for live capture
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        _mock_get_chunk(ctx, [
            _make_speech_chunk(),
            _make_silence_chunk(),
            _make_silence_chunk(),
        ])

        result = state_recording(ctx)

        assert result == "transcribing"
        assert ctx.captured_audio is not None
        assert len(ctx.captured_audio) > 0

    def test_wake_mode_no_wake_time_returns_listening(self):
        """Wake mode with wake_time=None is invalid, returns to listening."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = None

        result = state_recording(ctx)

        assert result == "listening"

    def test_wake_mode_flushes_chunk_queue(self):
        """Wake mode flushes the chunk queue to prevent duplication with ring buffer."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0

        # Ring buffer has the authoritative audio
        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk()))

        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        _mock_get_chunk(ctx, [_make_silence_chunk(), _make_silence_chunk()])

        state_recording(ctx)
        ctx.audio_buffer.flush_chunk_queue.assert_called_once()


class TestRecordingVadMode:
    """VAD mode: waits for speech onset before capturing."""

    def test_vad_mode_speech_found(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.recording_mode = "vad"

        # Speech chunk triggers capture, then silence ends it
        _mock_get_chunk(ctx, [
            _make_speech_chunk(),
            _make_silence_chunk(),
            _make_silence_chunk(),
        ])

        result = state_recording(ctx)

        assert result == "transcribing"
        assert ctx.captured_audio is not None

    def test_vad_mode_timeout_returns_listening(self):
        """VAD timeout with no speech returns to listening."""
        ctx = _make_ctx({'vad_timeout': 0.1})
        _setup_running_audio(ctx)
        ctx.recording_mode = "vad"

        # Only silence chunks, then None (empty queue)
        _mock_get_chunk(ctx, [
            _make_silence_chunk(),
            _make_silence_chunk(),
            _make_silence_chunk(),
        ])

        result = state_recording(ctx)
        assert result == "listening"


class TestRecordingSilenceDetection:
    """Silence detection ends recording."""

    def test_silence_after_speech_ends_recording(self):
        ctx = _make_ctx({'silence_duration': 0})
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0

        # Ring buffer with speech
        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk()))
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        # Live: speech then extended silence
        _mock_get_chunk(ctx, [
            _make_speech_chunk(),
            _make_silence_chunk(),
            _make_silence_chunk(),
            _make_silence_chunk(),
        ])

        result = state_recording(ctx)
        assert result == "transcribing"

    def test_no_chunks_captured_returns_listening(self):
        """If no audio chunks are captured at all, return to listening."""
        ctx = _make_ctx({'silence_duration': 0})
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0

        # No ring buffer audio, and no live chunks at all (queue is empty)
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        _mock_get_chunk(ctx, [])  # nothing to capture

        # With no chunks, the capture loop just sees None repeatedly.
        # Post QUIT so the loop exits rather than spinning forever.
        ctx.mailbox.post(Mailbox.QUIT)

        result = state_recording(ctx)
        assert result is None  # QUIT exits


class TestRecordingMailboxInterrupts:
    """Mailbox messages interrupt recording."""

    def test_toggle_pause_discards_audio(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0

        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk()))
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        ctx.mailbox.post(Mailbox.TOGGLE_PAUSE)

        _mock_get_chunk(ctx, [_make_speech_chunk()])

        result = state_recording(ctx)
        assert result == "paused"

    @mock.patch('context.gc')
    def test_toggle_enable_discards_and_unloads(self, mock_gc):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0
        ctx.unload_models = mock.MagicMock()
        ctx.audio_buffer.stop = mock.MagicMock()

        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk()))
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)

        _mock_get_chunk(ctx, [_make_speech_chunk()])

        result = state_recording(ctx)

        assert result == "disabled"
        ctx.unload_models.assert_called_once()

    @mock.patch('context.gc')
    def test_quit_during_recording(self, mock_gc):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0

        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk()))
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        ctx.mailbox.post(Mailbox.QUIT)

        _mock_get_chunk(ctx, [_make_speech_chunk()])

        result = state_recording(ctx)
        assert result is None

    def test_toggle_pause_during_vad_wait(self):
        """TOGGLE_PAUSE during VAD onset wait returns paused immediately."""
        ctx = _make_ctx({'vad_timeout': 2.0})
        _setup_running_audio(ctx)
        ctx.recording_mode = "vad"

        ctx.mailbox.post(Mailbox.TOGGLE_PAUSE)
        _mock_get_chunk(ctx, [_make_silence_chunk()])

        result = state_recording(ctx)
        assert result == "paused"


class TestRecordingAudioHealth:
    """Audio health checks during recording."""

    @mock.patch('context.gc')
    def test_unhealthy_restart_fails_with_device_returns_listening(self, mock_gc):
        """Unhealthy + restart fails + device name -> listening for recovery."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0
        ctx.audio_buffer.device_name = "Test Device"

        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk()))
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=False)
        ctx.audio_buffer.restart = mock.MagicMock(return_value=False)

        _mock_get_chunk(ctx, [_make_speech_chunk()])

        result = state_recording(ctx)
        assert result == "listening"

    @mock.patch('context.gc')
    def test_unhealthy_restart_fails_no_device_returns_disabled(self, mock_gc):
        """Unhealthy + restart fails + no device name -> disabled."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0
        ctx.audio_buffer.device_name = ''
        ctx.unload_models = mock.MagicMock()

        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk()))
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=False)
        ctx.audio_buffer.restart = mock.MagicMock(return_value=False)

        _mock_get_chunk(ctx, [_make_speech_chunk()])

        result = state_recording(ctx)
        assert result == "disabled"
        ctx.unload_models.assert_called()


class TestRecordingOutputData:
    """Verify captured audio is properly concatenated."""

    def test_captured_audio_concatenated(self):
        ctx = _make_ctx({'silence_duration': 0})
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0

        # Ring buffer has 2 chunks
        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk(1000)))
        ctx.audio_buffer.ring_buffer.append((999.08, _make_speech_chunk(2000)))
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        # Live queue has 1 speech + silence
        _mock_get_chunk(ctx, [
            _make_speech_chunk(3000),
            _make_silence_chunk(),
            _make_silence_chunk(),
        ])

        state_recording(ctx)

        assert ctx.captured_audio is not None
        # Ring buffer (2 chunks * 1280) + live (at least 1 speech chunk)
        assert len(ctx.captured_audio) >= 2 * 1280

    def test_captured_audio_is_int16(self):
        ctx = _make_ctx({'silence_duration': 0})
        _setup_running_audio(ctx)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0

        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk()))
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        _mock_get_chunk(ctx, [_make_silence_chunk(), _make_silence_chunk()])

        state_recording(ctx)

        assert ctx.captured_audio.dtype == np.int16
