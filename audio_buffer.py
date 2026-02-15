"""Atlas Voice audio buffer — continuous ring buffer with chunk queue."""

import os
import time
import wave
import queue
import threading

import numpy as np
import sounddevice as sd

from logging_utils import log_debug, log_info, log_error


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

        # Ring buffer: stores (timestamp, audio_chunk) tuples
        from collections import deque
        chunks_per_second = self.sample_rate / self.chunk_size
        max_chunks = int(config['buffer_seconds'] * chunks_per_second)
        self.ring_buffer = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

        # Chunk queue for wake word processing (separate from ring buffer)
        self.chunk_queue = queue.Queue(maxsize=1000)

        # Stream state
        self.stream = None
        self.last_callback_time = None
        self.current_amplitude = 0

    def start(self):
        """Start audio stream. Idempotent."""
        if self.stream is not None:
            return
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()
        log_info("[AUDIO] Stream started")

    def stop(self):
        """Stop stream. Idempotent, no-throw."""
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            log_info("[AUDIO] Stream stopped")
        except Exception as e:
            log_error(f"[AUDIO] Stop failed: {e}")
            self.stream = None

    def restart(self):
        """Stop and start with retry logic. Raises on failure after all retries."""
        retries = [1.0, 2.0, 4.0]
        for i, delay in enumerate(retries):
            try:
                self.stop()
                time.sleep(delay)
                self.start()
                log_info(f"[AUDIO] Restart successful (attempt {i+1})")
                return
            except Exception as e:
                log_error(f"[AUDIO] Restart attempt {i+1} failed: {e}")
        raise Exception("Audio stream restart failed after 3 attempts")

    def is_healthy(self):
        """Returns True if last callback within 2 seconds."""
        if self.last_callback_time is None:
            return False
        return (time.time() - self.last_callback_time) < 2.0

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
        if status:
            log_error(f"[AUDIO] Callback status: {status}")

        self.last_callback_time = time.time()

        # Convert float32 [-1, 1] to int16 [-32768, 32767]
        chunk = (indata[:, 0] * 32768).astype(np.int16)
        self.current_amplitude = np.abs(chunk).mean()

        timestamp = time.time()

        # Fill ring buffer (always)
        with self._lock:
            self.ring_buffer.append((timestamp, chunk.copy()))

        # Enqueue chunk for wake word processing (best effort)
        try:
            self.chunk_queue.put_nowait(chunk.copy())
        except queue.Full:
            log_debug("[AUDIO] Chunk queue full, dropping chunk")
