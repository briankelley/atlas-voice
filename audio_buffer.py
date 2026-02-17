"""Atlas Voice audio buffer — continuous ring buffer with chunk queue."""

import logging
import os
import time
import wave
import queue
import threading

import numpy as np
import sounddevice as sd

from logging_utils import log_debug, log_info, log_error

logger = logging.getLogger(__name__)


class AudioBuffer:
    """
    Continuously captures audio to a ring buffer.

    Ring buffer: circular buffer of recent audio history (configurable seconds).
    Chunk queue: secondary FIFO queue of chunks awaiting wake word processing.
    Flushing the chunk queue does NOT clear the ring buffer.
    """

    def __init__(self, config):
        self.sample_rate = config['sample_rate']
        self.chunk_size = config['chunk_size']
        self.silence_threshold = config['silence_threshold']

        # Device selection
        self.device_name = config.get('audio_device', '')
        self.device_index = None  # resolved fresh on every start()

        # Ring buffer: stores (timestamp, audio_chunk) tuples
        from collections import deque
        chunks_per_second = self.sample_rate / self.chunk_size
        max_chunks = int(config['buffer_seconds'] * chunks_per_second)
        self.ring_buffer = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

        # Chunk queue for wake word processing (separate from ring buffer)
        self.chunk_queue = queue.Queue(maxsize=100)  # bounded; ~8s at 80ms chunks

        # Stream state
        self.stream = None
        self.last_callback_time = None
        self.current_amplitude = 0

        # Error tracking (sliding window for disconnect detection)
        self._error_count_window = []       # list of error timestamps
        self._error_window_seconds = 5.0    # time window for error rate calculation
        self._error_rate_threshold = 0.8    # >80% error callbacks = unhealthy
        self._last_error_log_time = 0       # time-based log throttling
        self._first_callback_event = threading.Event()  # signaled on first clean callback
        self._error_lock = threading.Lock()

    def _resolve_device(self):
        """Resolve device name to sounddevice index.

        Returns int device index, or None if no name configured.
        Raises ValueError if name is configured but cannot be resolved.
        """
        if not self.device_name:
            return None

        devices = sd.query_devices()
        input_devices = [
            (i, d) for i, d in enumerate(devices)
            if d['max_input_channels'] > 0
        ]

        # Step 1: exact match (case-sensitive)
        for idx, dev in input_devices:
            if dev['name'] == self.device_name:
                log_info(f"[AUDIO] Device exact match: [{idx}] {dev['name']}")
                return idx

        # Step 2: case-insensitive substring match
        name_lower = self.device_name.lower()
        matches = [
            (idx, dev) for idx, dev in input_devices
            if name_lower in dev['name'].lower()
        ]

        # Step 3: multiple substring matches — ambiguous
        if len(matches) > 1:
            names = [dev['name'] for _, dev in matches]
            raise ValueError(
                f"Ambiguous audio device '{self.device_name}' matched {len(matches)} "
                f"devices: {names}. Use a more specific string."
            )

        # Single substring match
        if len(matches) == 1:
            idx, dev = matches[0]
            log_info(f"[AUDIO] Device substring match: [{idx}] {dev['name']}")
            return idx

        # Step 4: zero matches
        available_names = [dev['name'] for _, dev in input_devices]
        logger.debug(
            "[AUDIO] Full device list: %s",
            [(idx, dev) for idx, dev in input_devices]
        )
        raise ValueError(
            f"Audio device '{self.device_name}' not found. "
            f"Available input devices: {available_names}"
        )

    def is_device_present(self):
        """Check if the configured audio device is currently available.

        Returns True if the device can be resolved, False otherwise.
        Does NOT log on failure (caller manages logging).
        """
        try:
            self._resolve_device()
            return True
        except (ValueError, Exception):
            return False

    def start(self):
        """Start audio stream. Idempotent.

        Resolves device index fresh on each call. On any exception,
        calls stop() to ensure clean stopped state, then re-raises.
        Returns None on success.
        """
        if self.stream is not None:
            return
        try:
            self.device_index = self._resolve_device()

            # Reset error tracking state
            with self._error_lock:
                self._error_count_window = []
            self._last_error_log_time = 0
            self._first_callback_event.clear()

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self._audio_callback,
                blocksize=self.chunk_size,
                device=self.device_index
            )
            self.stream.start()
            log_info(
                f"[AUDIO] Stream started (device={self.device_index})"
            )
        except Exception:
            self.stop()
            raise

    def stop(self, force=False):
        """Stop stream. Idempotent, no-throw.

        Args:
            force: If True, use stream.abort() instead of stream.stop()
                   for more aggressive teardown (less likely to block
                   on vanished hardware).
        """
        if self.stream is None:
            self._first_callback_event.clear()
            return
        try:
            if force:
                self.stream.abort()
            else:
                self.stream.stop()
        except Exception as e:
            # Handle may be invalid if hardware vanished
            log_error(f"[AUDIO] Stream {'abort' if force else 'stop'} failed: {e}")
        try:
            self.stream.close()
        except Exception as e:
            log_error(f"[AUDIO] Stream close failed: {e}")
        self.stream = None
        self._first_callback_event.clear()
        log_info("[AUDIO] Stream stopped")

    def restart(self):
        """Stop and start. Returns True on success, False on failure.

        Never raises to callers. On failure, object is left in clean
        stopped state. No sleep, no retry loop.
        Uses abort() (force=True) since restart implies the stream is
        misbehaving — stream.stop() can block indefinitely on bad hardware.
        """
        try:
            self.stop(force=True)
            self.start()
            log_info("[AUDIO] Restart successful")
            return True
        except Exception as e:
            log_error(f"[AUDIO] Restart failed: {e}")
            return False

    def is_healthy(self):
        """Returns True if the audio stream is functioning normally.

        Three failure modes detected:
        1. Callback never fired (last_callback_time is None)
        2. Callback stopped firing (stale > 2s)
        3. Error rate exceeds threshold (KVM disconnect with PortAudio
           still invoking callback with error flags)
        """
        # Mode 1: callback never fired
        if self.last_callback_time is None:
            return False

        # Mode 2: callback stopped firing
        if (time.time() - self.last_callback_time) >= 2.0:
            return False

        # Mode 3: error rate in sliding window
        now = time.monotonic()
        with self._error_lock:
            cutoff = now - self._error_window_seconds
            errors_in_window = [t for t in self._error_count_window if t >= cutoff]
        chunk_duration = self.chunk_size / self.sample_rate
        expected_callbacks = self._error_window_seconds / chunk_duration
        if expected_callbacks > 0 and len(errors_in_window) > 0:
            error_rate = len(errors_in_window) / expected_callbacks
            if error_rate > self._error_rate_threshold:
                return False

        return True

    def get_chunk(self, timeout=0.08):
        """Get next chunk from queue for wake word processing."""
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush_chunk_queue(self):
        """Discard all pending chunks in processing queue (NOT the ring buffer)."""
        count = 0
        try:
            while True:
                self.chunk_queue.get_nowait()
                count += 1
        except queue.Empty:
            pass
        if count > 0:
            log_debug(f"[AUDIO] Chunk queue flushed ({count} chunks)")

    def get_audio_since(self, timestamp):
        """Get all audio chunks since timestamp. Returns concatenated int16 numpy array."""
        chunks = []
        with self._lock:
            for ts, chunk in self.ring_buffer:
                if ts >= timestamp:
                    chunks.append(chunk)
        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.int16)

    def has_audio_since(self, timestamp):
        """Check if timestamp is still in ring buffer (not overwritten)."""
        with self._lock:
            if not self.ring_buffer:
                return False
            oldest_ts = self.ring_buffer[0][0]
            return timestamp >= oldest_ts

    def detect_speech_during(self, start_time, end_time):
        """Check if speech happened during time range. Returns timestamp of speech onset, or None."""
        with self._lock:
            for ts, chunk in self.ring_buffer:
                if ts >= start_time and ts <= end_time:
                    if np.abs(chunk).mean() >= self.silence_threshold:
                        return ts
        return None

    def get_speech_start_time(self, lookback_seconds=5.0):
        """Find when speech started within the lookback window. Returns timestamp or None."""
        cutoff = time.time() - lookback_seconds
        with self._lock:
            for ts, chunk in self.ring_buffer:
                if ts >= cutoff and np.abs(chunk).mean() >= self.silence_threshold:
                    return ts
        return None

    def save_to_wav(self, audio_data, filepath):
        """Save int16 audio data to a WAV file."""
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice in audio thread."""
        self.last_callback_time = time.time()

        try:
            if status:
                # Error callback: track in sliding window
                with self._error_lock:
                    self._error_count_window.append(time.monotonic())

                # Throttled error logging (at most once per 5 seconds)
                now = time.monotonic()
                if (now - self._last_error_log_time) > 5.0:
                    log_error(f"[AUDIO] Callback status: {status}")
                    self._last_error_log_time = now

                # Skip enqueue — do not process error-status frames
                return

            # Clean callback: prune old errors from sliding window
            with self._error_lock:
                cutoff = time.monotonic() - self._error_window_seconds
                self._error_count_window = [
                    t for t in self._error_count_window if t >= cutoff
                ]

            # Signal first successful callback
            if not self._first_callback_event.is_set():
                self._first_callback_event.set()

            # Convert float32 [-1, 1] to int16 [-32768, 32767]
            chunk = (indata[:, 0] * 32768).astype(np.int16)
            self.current_amplitude = np.abs(chunk).mean()

            timestamp = time.time()

            # Fill ring buffer (always)
            with self._lock:
                self.ring_buffer.append((timestamp, chunk.copy()))

            # Enqueue chunk for wake word processing (overflow: discard oldest)
            try:
                self.chunk_queue.put_nowait(chunk.copy())
            except queue.Full:
                try:
                    self.chunk_queue.get_nowait()  # discard oldest
                except queue.Empty:
                    pass
                self.chunk_queue.put_nowait(chunk.copy())

        except Exception as e:
            # Treat callback exceptions as disconnect signal
            log_error(f"[AUDIO] Callback exception: {e}")
            with self._error_lock:
                self._error_count_window.append(time.monotonic())
