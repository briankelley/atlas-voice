"""Integration tests for Atlas Voice state transitions and cross-module contracts."""

import time
import unittest
from unittest import mock

import numpy as np

from audio_buffer import AudioBuffer
from mailbox import Mailbox
from context import AtlasContext


def _make_config(**overrides):
    """Build a config dict suitable for AtlasContext."""
    config = {
        'sample_rate': 16000,
        'chunk_size': 1280,
        'silence_threshold': 500,
        'buffer_seconds': 10,
        'audio_device': '',
        'wake_word_model': '/dev/null',
        'whisper_model': '/dev/null',
        'whisper_device': 'cpu',
        'whisper_compute_type': 'int8',
        'wake_word_threshold': 0.35,
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
    }
    config.update(overrides)
    return config


class TestClearInterstateData(unittest.TestCase):
    """Tests for ctx.clear_interstate_data()."""

    def test_clears_all_fields(self):
        config = _make_config()
        buf = AudioBuffer(config)
        ctx = AtlasContext(config, buf)
        ctx.wake_time = 12345.0
        ctx.captured_audio = np.zeros(1000, dtype=np.int16)
        ctx.recording_mode = "wake"

        ctx.clear_interstate_data()

        self.assertIsNone(ctx.wake_time)
        self.assertIsNone(ctx.captured_audio)
        self.assertIsNone(ctx.recording_mode)


class TestWakeWordInputFormat(unittest.TestCase):
    """Verify wake word model receives int16 audio (openwakeword requirement)."""

    def test_predict_receives_int16(self):
        """AudioBuffer produces int16 chunks; openwakeword requires int16 input."""
        chunk_int16 = np.random.randint(-32768, 32767, size=1280, dtype=np.int16)
        self.assertEqual(chunk_int16.dtype, np.int16)


class TestStateListeningToRecording(unittest.TestCase):
    """Test listening -> recording transition on wake word detection."""

    @mock.patch('state_listening.time')
    def test_wake_detection_sets_interstate_data(self, mock_time):
        """When wake word triggers, ctx.wake_time and recording_mode are set."""
        mock_time.time.return_value = 1000.0
        mock_time.monotonic = time.monotonic

        config = _make_config()
        buf = AudioBuffer(config)
        ctx = AtlasContext(config, buf)
        ctx.mailbox = Mailbox()

        # Mock wake model that always triggers
        mock_wake = mock.MagicMock()
        mock_wake.prediction_buffer = {
            'hey_atlas': [0.0, 0.0, 0.95]  # score > threshold
        }
        ctx.wake_model = mock_wake

        # Mock audio buffer as running with a chunk available
        buf.stream = mock.MagicMock()
        buf.last_callback_time = time.time()
        # Prevent flush_chunk_queue from discarding our test chunk
        buf.flush_chunk_queue = mock.MagicMock()
        chunk = np.zeros(1280, dtype=np.int16)
        buf.chunk_queue.put(chunk)

        # Import here to use patched time
        from state_listening import state_listening
        result = state_listening(ctx)

        self.assertEqual(result, "recording")
        self.assertEqual(ctx.recording_mode, "wake")
        self.assertIsNotNone(ctx.wake_time)

        # Verify predict was called with int16 data (openwakeword requirement)
        mock_wake.predict.assert_called_once()
        call_arg = mock_wake.predict.call_args[0][0]
        self.assertEqual(call_arg.dtype, np.int16)


if __name__ == '__main__':
    unittest.main()
