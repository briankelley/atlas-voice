"""Unit tests for AudioBuffer — mocks sounddevice to avoid real hardware."""

import queue
import time
import unittest
from unittest import mock

from audio_buffer import AudioBuffer


def _make_config(**overrides):
    """Build a minimal config dict suitable for AudioBuffer.__init__."""
    config = {
        'sample_rate': 16000,
        'chunk_size': 1280,
        'silence_threshold': 500,
        'buffer_seconds': 10,
        'audio_device': '',
    }
    config.update(overrides)
    return config


def _make_device(name, max_input_channels=1, max_output_channels=0):
    """Return a dict mimicking sounddevice.query_devices() entries."""
    return {
        'name': name,
        'max_input_channels': max_input_channels,
        'max_output_channels': max_output_channels,
    }


class TestResolveDevice(unittest.TestCase):
    """Tests for AudioBuffer._resolve_device()."""

    def _make_buffer(self, device_name):
        cfg = _make_config(audio_device=device_name)
        return AudioBuffer(cfg)

    @mock.patch('audio_buffer.sd')
    def test_resolve_device_exact_match(self, mock_sd):
        """Exact name match returns the correct device index."""
        mock_sd.query_devices.return_value = [
            _make_device('HDA Intel PCH: ALC892', max_input_channels=2),
            _make_device('USB Audio Device', max_input_channels=1),
            _make_device('HDMI Output', max_input_channels=0, max_output_channels=8),
        ]
        buf = self._make_buffer('USB Audio Device')
        idx = buf._resolve_device()
        self.assertEqual(idx, 1)

    @mock.patch('audio_buffer.sd')
    def test_resolve_device_substring_match(self, mock_sd):
        """Case-insensitive substring fallback returns correct index."""
        mock_sd.query_devices.return_value = [
            _make_device('HDA Intel PCH: ALC892', max_input_channels=2),
            _make_device('Samson GoMic USB Microphone', max_input_channels=1),
            _make_device('HDMI Output', max_input_channels=0, max_output_channels=8),
        ]
        buf = self._make_buffer('gomic usb')
        idx = buf._resolve_device()
        self.assertEqual(idx, 1)

    @mock.patch('audio_buffer.sd')
    def test_resolve_device_multi_match_error(self, mock_sd):
        """Ambiguous substring matching two devices raises ValueError."""
        mock_sd.query_devices.return_value = [
            _make_device('USB Audio Device A', max_input_channels=1),
            _make_device('USB Audio Device B', max_input_channels=1),
        ]
        buf = self._make_buffer('usb audio')
        with self.assertRaises(ValueError) as ctx:
            buf._resolve_device()
        msg = str(ctx.exception)
        self.assertIn('USB Audio Device A', msg)
        self.assertIn('USB Audio Device B', msg)

    @mock.patch('audio_buffer.sd')
    def test_resolve_device_no_match_error(self, mock_sd):
        """No match raises ValueError listing available input devices."""
        mock_sd.query_devices.return_value = [
            _make_device('HDA Intel PCH', max_input_channels=2),
            _make_device('Webcam Mic', max_input_channels=1),
        ]
        buf = self._make_buffer('Nonexistent XYZ')
        with self.assertRaises(ValueError) as ctx:
            buf._resolve_device()
        msg = str(ctx.exception)
        self.assertIn('HDA Intel PCH', msg)
        self.assertIn('Webcam Mic', msg)

    @mock.patch('audio_buffer.sd')
    def test_resolve_device_empty_name(self, mock_sd):
        """Empty device_name returns None without querying devices."""
        buf = self._make_buffer('')
        result = buf._resolve_device()
        self.assertIsNone(result)
        mock_sd.query_devices.assert_not_called()

    @mock.patch('audio_buffer.sd')
    def test_resolve_device_ignores_output_only(self, mock_sd):
        """Output-only device (max_input_channels=0) is never selected."""
        mock_sd.query_devices.return_value = [
            _make_device('HDMI Speaker', max_input_channels=0, max_output_channels=8),
        ]
        buf = self._make_buffer('HDMI Speaker')
        with self.assertRaises(ValueError):
            buf._resolve_device()


class TestIsHealthy(unittest.TestCase):
    """Tests for AudioBuffer.is_healthy() error-rate logic."""

    def _make_buffer(self):
        cfg = _make_config()
        return AudioBuffer(cfg)

    def test_is_healthy_error_rate(self):
        """High error rate within the sliding window makes is_healthy() False."""
        buf = self._make_buffer()
        buf.last_callback_time = time.time()  # recent, so Mode 1/2 pass

        # chunk_duration = 1280 / 16000 = 0.08s
        # expected_callbacks in 5s window = 5.0 / 0.08 = 62.5
        # 80% threshold = 50 errors needed
        # Populate with 55 error timestamps spread across recent window
        now = time.monotonic()
        buf._error_count_window = [now - i * 0.05 for i in range(55)]

        self.assertFalse(buf.is_healthy())

    def test_is_healthy_transient_errors(self):
        """A few errors below the 80% threshold keeps is_healthy() True."""
        buf = self._make_buffer()
        buf.last_callback_time = time.time()  # recent

        # Only 5 errors in window — well below 50 needed for 80%
        now = time.monotonic()
        buf._error_count_window = [now - i * 0.5 for i in range(5)]

        self.assertTrue(buf.is_healthy())


class TestQueueOverflow(unittest.TestCase):
    """Tests for bounded chunk_queue overflow behavior."""

    def test_queue_bounded_overflow(self):
        """Overflow drops the oldest item, queue stays at maxsize."""
        cfg = _make_config()
        buf = AudioBuffer(cfg)

        # Fill queue to capacity (maxsize=100) with (timestamp, value) tuples
        for i in range(100):
            buf.chunk_queue.put_nowait((float(i), i))
        self.assertEqual(buf.chunk_queue.qsize(), 100)

        # Simulate the callback overflow logic: drop oldest tuple, then put new
        newest = 999
        try:
            buf.chunk_queue.put_nowait((float(newest), newest))
        except queue.Full:
            buf.chunk_queue.get_nowait()   # discard oldest tuple
            buf.chunk_queue.put_nowait((float(newest), newest))

        self.assertEqual(buf.chunk_queue.qsize(), 100)

        # Drain the queue and verify the newest item is present
        items = []
        while not buf.chunk_queue.empty():
            ts, val = buf.chunk_queue.get_nowait()
            items.append(val)
        self.assertIn(newest, items)
        # The very first item (0) should have been dropped
        self.assertNotIn(0, items)


class TestStartAndRestart(unittest.TestCase):
    """Tests for start() failure cleanup and restart() return values."""

    @mock.patch('audio_buffer.sd')
    def test_start_failure_leaves_clean_state(self, mock_sd):
        """start() failure leaves stream as None (clean stopped state)."""
        mock_sd.InputStream.side_effect = RuntimeError('No audio device')
        cfg = _make_config()
        buf = AudioBuffer(cfg)

        with self.assertRaises(RuntimeError):
            buf.start()

        self.assertIsNone(buf.stream)

    @mock.patch('audio_buffer.sd')
    def test_restart_returns_bool(self, mock_sd):
        """restart() returns True on success, False on failure."""
        mock_stream = mock.MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        cfg = _make_config()
        buf = AudioBuffer(cfg)

        # Successful restart
        result = buf.restart()
        self.assertTrue(result)

        # Now make InputStream raise on the next start()
        buf.stream = None  # ensure stop() is a no-op path
        mock_sd.InputStream.side_effect = RuntimeError('Device vanished')

        result = buf.restart()
        self.assertFalse(result)


class TestGetChunkTimestamp(unittest.TestCase):
    """Tests for get_chunk() timestamp storage via last_dequeued_ts."""

    def _make_buffer(self):
        cfg = _make_config()
        return AudioBuffer(cfg)

    def test_get_chunk_returns_chunk_and_stores_timestamp(self):
        """get_chunk() unpacks (ts, chunk), returns chunk, stores ts."""
        import numpy as np
        buf = self._make_buffer()
        chunk = np.zeros(1280, dtype=np.int16)
        ts = 12345.678
        buf.chunk_queue.put_nowait((ts, chunk))

        result = buf.get_chunk()
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, np.int16)
        self.assertEqual(len(result), 1280)
        self.assertEqual(buf.last_dequeued_ts, ts)

    def test_get_chunk_returns_none_on_empty(self):
        """get_chunk() returns None on empty queue, last_dequeued_ts unchanged."""
        buf = self._make_buffer()
        buf.last_dequeued_ts = 99999.0  # pre-set

        result = buf.get_chunk(timeout=0.01)
        self.assertIsNone(result)
        self.assertEqual(buf.last_dequeued_ts, 99999.0)

    def test_flush_resets_last_dequeued_ts(self):
        """flush_chunk_queue() sets last_dequeued_ts to None."""
        import numpy as np
        buf = self._make_buffer()
        chunk = np.zeros(1280, dtype=np.int16)
        buf.chunk_queue.put_nowait((1000.0, chunk))
        buf.get_chunk()
        self.assertEqual(buf.last_dequeued_ts, 1000.0)

        buf.flush_chunk_queue()
        self.assertIsNone(buf.last_dequeued_ts)


class TestOverflowTupleHandling(unittest.TestCase):
    """Tests that overflow path correctly handles (timestamp, chunk) tuples."""

    def test_overflow_preserves_tuple_contract(self):
        """After overflow, get_chunk() returns bare chunk with correct timestamp."""
        import numpy as np
        cfg = _make_config()
        buf = AudioBuffer(cfg)

        # Fill queue to capacity with tuples
        for i in range(100):
            chunk = np.full(1280, i, dtype=np.int16)
            buf.chunk_queue.put_nowait((float(i), chunk))
        self.assertEqual(buf.chunk_queue.qsize(), 100)

        # Trigger overflow with a new tuple
        new_ts = 999.0
        new_chunk = np.full(1280, 42, dtype=np.int16)
        try:
            buf.chunk_queue.put_nowait((new_ts, new_chunk))
        except queue.Full:
            buf.chunk_queue.get_nowait()  # discard oldest tuple
            buf.chunk_queue.put_nowait((new_ts, new_chunk))

        # Drain via get_chunk() — should return bare chunks, not tuples
        chunks = []
        timestamps = []
        while True:
            c = buf.get_chunk(timeout=0.01)
            if c is None:
                break
            chunks.append(c)
            timestamps.append(buf.last_dequeued_ts)

        # All returned items should be numpy arrays, not tuples
        for c in chunks:
            self.assertEqual(c.dtype, np.int16)

        # Last dequeued timestamp should be our overflow item's timestamp
        self.assertEqual(timestamps[-1], new_ts)


if __name__ == '__main__':
    unittest.main()
