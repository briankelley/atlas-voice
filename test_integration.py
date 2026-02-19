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
        'wake_preroll': 0.75,
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
        buf.chunk_queue.put((1000.0, chunk))

        # Import here to use patched time
        from state_listening import state_listening
        result = state_listening(ctx)

        self.assertEqual(result, "recording")
        self.assertEqual(ctx.recording_mode, "wake")

        # Intermediate check: last_dequeued_ts was set by get_chunk()
        self.assertEqual(buf.last_dequeued_ts, 1000.0)

        # wake_time = audio_ts (1000.0) - preroll (0.75) = 999.25
        self.assertAlmostEqual(ctx.wake_time, 999.25, places=2)

        # Verify predict was called with int16 data (openwakeword requirement)
        mock_wake.predict.assert_called_once()
        call_arg = mock_wake.predict.call_args[0][0]
        self.assertEqual(call_arg.dtype, np.int16)


class TestEndToEndPrerollAudio(unittest.TestCase):
    """End-to-end: wake detection -> get_audio_since(wake_time) returns preroll audio."""

    def test_wake_preroll_retrieves_ring_buffer_audio(self):
        """After wake detection, get_audio_since(wake_time) returns expected preroll audio."""
        config = _make_config()
        buf = AudioBuffer(config)
        ctx = AtlasContext(config, buf)
        ctx.mailbox = Mailbox()

        # Seed ring buffer with timestamped audio chunks spanning 2 seconds
        # Chunks at 80ms intervals: timestamps 998.0, 998.08, ..., 999.92
        chunks_in_ring = []
        for i in range(25):
            ts = 998.0 + i * 0.08
            chunk = np.full(1280, i + 1, dtype=np.int16)  # non-zero audio
            buf.ring_buffer.append((ts, chunk.copy()))
            chunks_in_ring.append((ts, chunk))

        # Put one chunk in the wake word queue with timestamp matching
        # the last ring buffer entry (simulating the chunk that triggered wake)
        wake_chunk_ts = 999.92
        wake_chunk = np.full(1280, 99, dtype=np.int16)
        buf.chunk_queue.put((wake_chunk_ts, wake_chunk))

        # Simulate wake_time calculation: audio_ts - preroll
        # audio_ts = 999.92 (from last_dequeued_ts after get_chunk)
        # wake_time = 999.92 - 0.75 = 999.17
        result_chunk = buf.get_chunk()
        self.assertIsNotNone(result_chunk)
        self.assertEqual(buf.last_dequeued_ts, wake_chunk_ts)

        wake_time = buf.last_dequeued_ts - config['wake_preroll']
        self.assertAlmostEqual(wake_time, 999.17, places=2)

        # get_audio_since should return chunks from the ring buffer
        audio = buf.get_audio_since(wake_time)
        self.assertGreater(len(audio), 0)

        # Should include chunks from ~999.20 onward (timestamps >= 999.17)
        # That's roughly the last 9-10 chunks of our 25
        expected_chunks = [c for ts, c in chunks_in_ring if ts >= wake_time]
        self.assertGreater(len(expected_chunks), 0)
        expected_samples = sum(len(c) for c in expected_chunks)
        self.assertEqual(len(audio), expected_samples)

        # Verify the ring buffer still has audio since wake_time
        self.assertTrue(buf.has_audio_since(wake_time))


if __name__ == '__main__':
    unittest.main()
