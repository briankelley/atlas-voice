#!/usr/bin/env python3
"""
Atlas - Always-listening voice dictation with wake word detection.

Uses OpenWakeWord for wake word detection, faster-whisper for transcription.
Say "Hey Atlas" to start dictating.

Features:
- Continuous audio buffer - never misses audio, even during transcription
- Wake word activation
- Automatic silence detection
- Spoken punctuation and word replacement
- System tray icon for enable/disable control

Usage: python3 atlas.py
"""

import subprocess
import tempfile
import threading
import signal
import wave
import sys
import os
import time
import re
from collections import deque

# Ensure DISPLAY is set (needed when launched from environments without it)
if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'

# CUDA runtime libraries live in the venv (nvidia-cublas-cu12, nvidia-cudnn-cu12).
# LD_LIBRARY_PATH must be set before the process starts, so re-exec if needed.
_venv_nvidia = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv",
                            "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                            "site-packages", "nvidia")
if os.path.isdir(_venv_nvidia) and '_ATLAS_CUDA_READY' not in os.environ:
    _cuda_lib_paths = [os.path.join(_venv_nvidia, d, "lib") for d in os.listdir(_venv_nvidia)
                       if os.path.isdir(os.path.join(_venv_nvidia, d, "lib"))]
    if _cuda_lib_paths:
        _existing = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = ':'.join(_cuda_lib_paths + ([_existing] if _existing else []))
        os.environ['_ATLAS_CUDA_READY'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('AyatanaAppIndicator3', '0.1')
from gi.repository import Gtk, AyatanaAppIndicator3, GLib

import numpy as np
import sounddevice as sd
from openwakeword.model import Model as WakeWordModel
from faster_whisper import WhisperModel

# ============================================================================
# Configuration
# ============================================================================

# Base directory (where this script lives)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Wake word model - custom trained "Hey Atlas"
WAKE_WORD_MODEL = os.path.join(_BASE_DIR, "models", "openwakeword", "hey_atlas.tflite")
WAKE_WORD_THRESHOLD = 0.35  # Detection confidence threshold (lower = more sensitive)

# Whisper settings
WHISPER_MODEL = os.path.join(_BASE_DIR, "models", "whisper-large-v3")
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms chunks for wake word (16000 * 0.08)
BUFFER_SECONDS = 120  # Keep 2 minutes of audio in buffer
SILENCE_THRESHOLD = 500  # Amplitude threshold for silence detection
SILENCE_DURATION = 2.0  # Seconds of silence to stop recording
MAX_RECORD_DURATION = 60  # Max seconds to record per chunk

# Session settings
BREAK_PHRASE = "break break"  # Say this to end session and hit Enter

# Behavior
AUTO_TYPE = True
BEEP_ON_WAKE = True
DEBUG_MODE = False  # Print debug info (set True for troubleshooting)

# Tray icon
TRAY_ENABLED = True  # Set False to disable system tray icon

# ============================================================================
# Spoken punctuation & word replacements
# ============================================================================

# Aggressive punctuation replacement - if you say it, you mean the symbol
# Order matters: longer phrases first, then single words
SPOKEN_PUNCTUATION = [
    # Multi-word phrases first
    (r',?\s*\bopen parenthesis\b[,.]?\s*', ' ('),
    (r',?\s*\bclose parenthesis\b[,.]?\s*', ') '),
    (r',?\s*\bopen paren\b[,.]?\s*', ' ('),
    (r',?\s*\bclose paren\b[,.]?\s*', ') '),
    (r',?\s*\bleft paren\b[,.]?\s*', ' ('),
    (r',?\s*\bright paren\b[,.]?\s*', ') '),
    (r',?\s*\bopen quote\b[,.]?\s*', ' "'),
    (r',?\s*\bclose quote\b[,.]?\s*', '" '),
    (r',?\s*\bend quote\b[,.]?\s*', '" '),
    (r',?\s*\bopen bracket\b[,.]?\s*', ' ['),
    (r',?\s*\bclose bracket\b[,.]?\s*', '] '),
    (r',?\s*\bleft bracket\b[,.]?\s*', ' ['),
    (r',?\s*\bright bracket\b[,.]?\s*', '] '),
    (r',?\s*\bopen brace\b[,.]?\s*', ' {'),
    (r',?\s*\bclose brace\b[,.]?\s*', '} '),
    (r',?\s*\bexclamation point\b[,.]?\s*', '! '),
    (r',?\s*\bexclamation mark\b[,.]?\s*', '! '),
    (r',?\s*\bquestion mark\b[,.]?\s*', '? '),
    (r',?\s*\bnew paragraph\b[,.]?\s*', '\n\n'),
    (r',?\s*\bnew line\b[,.]?\s*', '\n'),
    (r',?\s*\bdot dot dot\b[,.]?\s*', '... '),
    # Single words - aggressive replacement, no conditions
    (r',?\s*\bnewline\b[,.]?\s*', '\n'),
    (r',?\s*\bperiod\b[,.]?\s*', '. '),
    (r',?\s*\bcomma\b[,.]?\s*', ', '),
    (r',?\s*\bcolon\b[,.]?\s*', ': '),
    (r',?\s*\bsemicolon\b[,.]?\s*', '; '),
    (r',?\s*\bdash\b[,.]?\s*', ' - '),
    (r',?\s*\bhyphen\b[,.]?\s*', '-'),
    (r',?\s*\bunquote\b[,.]?\s*', '" '),
    (r',?\s*\bellipsis\b[,.]?\s*', '... '),
    (r',?\s*\bapostrophe\b[,.]?\s*', "'"),
    (r',?\s*\bampersand\b[,.]?\s*', ' & '),
    (r',?\s*\basterisk\b[,.]?\s*', '*'),
    (r',?\s*\bat sign\b[,.]?\s*', '@'),
    (r',?\s*\bhashtag\b[,.]?\s*', '#'),
    # (r',?\s*\bpercent\b[,.]?\s*', '%'),
    (r',?\s*\bdollar sign\b[,.]?\s*', '$'),
    (r',?\s*\bbackslash\b[,.]?\s*', '\\\\'),
    (r',?\s*\bforward slash\b[,.]?\s*', '/'),
    (r',?\s*\bslash\b[,.]?\s*', '/'),
    (r',?\s*\bunderscore\b[,.]?\s*', '_'),
    (r',?\s*\bequals\b[,.]?\s*', ' = '),
    (r',?\s*\bplus\b[,.]?\s*', ' + '),
    (r',?\s*\bminus\b[,.]?\s*', ' - '),
]

WORD_REPLACEMENTS = {
    'cloud': 'Claude',
    'clawed': 'Claude',
    'claud': 'Claude',
    'clod': 'Claude',
    'club': 'Claude',
    'Cloud': 'Claude',
    'Clod': 'Claude',
    'Club': 'Claude',
    'in gram': 'engram',
    'In gram': 'engram',
    'ingram': 'engram',
    'Ingram': 'engram',
    'en gram': 'engram',
    'En gram': 'engram',
    'end gram': 'engram',
    'End gram': 'engram',
    'and gram': 'engram',
    'And gram': 'engram',
    'n gram': 'engram',
    'N gram': 'engram',
    'pseudo': 'sudo',
    'Pseudo': 'sudo',
    'no help': 'nohup',
    'no hup': 'nohup',
    'no hub': 'nohup',
    'no hop': 'nohup',
    'No help': 'nohup',
    'No hup': 'nohup',
    'thank you': '',
    'Thank you': '',
    'Thank You': '',
}


def process_text(text):
    """Apply punctuation conversion and word replacements."""
    for pattern, replacement in SPOKEN_PUNCTUATION:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    for wrong, right in WORD_REPLACEMENTS.items():
        text = re.sub(r'\b' + wrong + r'\b', right, text)

    # Clean up whitespace only - preserve duplicate punctuation as bonus context
    text = re.sub(r' +', ' ', text)                    # Multiple spaces → single
    text = re.sub(r' ([.,!?;:\)\]\}])', r'\1', text)   # Remove space before closing punct
    text = re.sub(r'([(\[\{]) ', r'\1', text)          # Remove space after opening punct

    return text.strip()


# ============================================================================
# Audio Buffer - Continuous capture with ring buffer
# ============================================================================

class AudioBuffer:
    """
    Continuously captures audio to a ring buffer.

    This allows us to:
    1. Never miss audio - even during transcription
    2. Extract audio segments by timestamp
    3. Check if speech occurred during any time window
    """

    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, buffer_seconds=BUFFER_SECONDS):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Calculate buffer size
        chunks_per_second = sample_rate / chunk_size
        max_chunks = int(buffer_seconds * chunks_per_second)

        # Ring buffer: stores (timestamp, audio_chunk) tuples
        self.buffer = deque(maxlen=max_chunks)
        self.lock = threading.Lock()

        # Current state
        self.running = False
        self.stream = None
        self.current_amplitude = 0
        self.callback_count = 0

        # Queue for sequential chunk processing (wake word needs every chunk in order)
        self.chunk_queue = deque(maxlen=1000)  # ~80 seconds of backlog max

    def start(self):
        """Start continuous audio capture."""
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()

    def stop(self):
        """Stop audio capture."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Called for each audio chunk from sounddevice."""
        self.callback_count += 1

        # Convert float32 [-1, 1] to int16 [-32768, 32767]
        chunk = (indata[:, 0] * 32768).astype(np.int16)
        self.current_amplitude = np.abs(chunk).mean()

        timestamp = time.time()
        with self.lock:
            self.buffer.append((timestamp, chunk.copy()))
            # Also queue for sequential wake word processing
            self.chunk_queue.append(chunk.copy())

    def get_amplitude(self):
        """Get current audio amplitude."""
        return self.current_amplitude

    def get_latest_chunk(self):
        """Get most recent audio chunk for wake word detection."""
        with self.lock:
            if self.buffer:
                return self.buffer[-1][1]
        return np.zeros(self.chunk_size, dtype=np.int16)

    def get_next_chunk(self):
        """
        Get next chunk from queue for sequential processing.
        Returns None if no chunks available.
        Wake word detection needs chunks fed in order, not just 'latest'.
        """
        with self.lock:
            if self.chunk_queue:
                return self.chunk_queue.popleft()
        return None

    def chunks_available(self):
        """Return number of chunks waiting in queue."""
        with self.lock:
            return len(self.chunk_queue)

    def flush_chunks(self):
        """Clear all pending chunks from queue. Call after session end to prevent re-trigger."""
        with self.lock:
            self.chunk_queue.clear()

    def get_audio_since(self, start_time):
        """
        Get all audio chunks since a given timestamp.
        Returns concatenated audio as int16 numpy array.
        """
        chunks = []
        with self.lock:
            for ts, chunk in self.buffer:
                if ts >= start_time:
                    chunks.append(chunk)

        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.int16)

    def has_speech_since(self, start_time, threshold=SILENCE_THRESHOLD):
        """Check if there's been speech since a given timestamp."""
        with self.lock:
            for ts, chunk in self.buffer:
                if ts >= start_time and np.abs(chunk).mean() >= threshold:
                    return True
        return False

    def get_speech_start_time(self, lookback_seconds=5.0, threshold=SILENCE_THRESHOLD):
        """
        Find when speech started within the lookback window.
        Returns timestamp of first speech, or None if no speech.
        """
        cutoff = time.time() - lookback_seconds

        with self.lock:
            for ts, chunk in self.buffer:
                if ts >= cutoff and np.abs(chunk).mean() >= threshold:
                    return ts
        return None

    def save_to_wav(self, audio_data, filepath):
        """Save int16 audio data to a WAV file."""
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())


# ============================================================================
# Audio Utilities
# ============================================================================

def play_beep():
    """Play a short beep sound."""
    if not BEEP_ON_WAKE:
        return
    try:
        subprocess.run(
            ["paplay", "/usr/share/sounds/freedesktop/stereo/audio-volume-change.oga"],
            capture_output=True,
            timeout=1
        )
    except:
        pass


# ============================================================================
# System Tray Icon
# ============================================================================

class TrayIcon:
    """
    System tray icon for Atlas control using AyatanaAppIndicator3.

    States:
    - ON: Listening for wake word
    - OFF: Paused (models still loaded)
    - RECORDING: Actively capturing/transcribing
    - DISABLE: Models unloaded, GPU released
    """

    STATE_ON = "ON"
    STATE_OFF = "OFF"
    STATE_RECORDING = "RECORDING"
    STATE_DISABLE = "DISABLE"

    def __init__(self, atlas_instance):
        self.atlas = atlas_instance
        self.state = self.STATE_ON
        self._paused = threading.Event()  # Clear = paused (OFF), Set = listening (ON)
        self._paused.set()  # Start in ON state (listening)
        self._indicator = None
        self._icon_dir = os.path.join(_BASE_DIR, "icons")

    def set_state(self, state):
        """Update the tray icon to reflect a new state."""
        self.state = state
        if self._indicator:
            icon_name = f"AV_{state}"
            GLib.idle_add(self._indicator.set_icon_full, icon_name, state)

    def is_listening(self):
        """Return True if in ON state (listening for wake word)."""
        return self._paused.is_set() and self.state != self.STATE_DISABLE

    def is_models_loaded(self):
        """Return True if models are loaded (not in DISABLE state)."""
        return self.state != self.STATE_DISABLE

    def _on_left_click(self, _):
        """Left-click: toggle ON/OFF (ignored when DISABLE)."""
        if self.state == self.STATE_DISABLE:
            # Ignored when disabled
            return
        if self._paused.is_set():
            # Currently ON -> go to OFF
            self._paused.clear()
            self.set_state(self.STATE_OFF)
            print("\n[Tray] Atlas paused (OFF)")
        else:
            # Currently OFF -> go to ON
            self._paused.set()
            self.set_state(self.STATE_ON)
            print("\n[Tray] Atlas resumed (ON)")

    def _on_enable_disable(self, _):
        """Menu: Enable/Disable (load/unload models)."""
        if self.state == self.STATE_DISABLE:
            # Currently disabled -> enable (load models)
            print("\n[Tray] Enabling Atlas (loading models)...")
            # Restart audio buffer to ensure fresh stream
            self.atlas.restart_audio()
            self.atlas.load_models()
            self._paused.set()
            self.set_state(self.STATE_ON)
            print("[Tray] Atlas enabled (ON)")
        else:
            # Currently enabled -> disable (unload models)
            print("\n[Tray] Disabling Atlas (unloading models)...")
            self._paused.clear()
            self.atlas.unload_models()
            # Stop audio buffer to fully release resources
            self.atlas.stop_audio()
            self.set_state(self.STATE_DISABLE)
            print("[Tray] Atlas disabled (models unloaded)")

    def _on_quit(self, _):
        print("\n[Tray] Quit requested")
        self.atlas.stop()
        GLib.idle_add(Gtk.main_quit)
        # Send SIGTERM to ourselves for clean systemd-aware shutdown
        threading.Thread(target=lambda: (time.sleep(1), os.kill(os.getpid(), signal.SIGTERM)), daemon=True).start()

    def _build_menu(self):
        menu = Gtk.Menu()

        # Pause/Resume (ON/OFF toggle)
        pause_item = Gtk.MenuItem(label="Pause / Resume")
        pause_item.connect("activate", self._on_left_click)
        menu.append(pause_item)

        # Enable/Disable (load/unload models)
        enable_item = Gtk.MenuItem(label="Enable / Disable (GPU)")
        enable_item.connect("activate", self._on_enable_disable)
        menu.append(enable_item)

        sep = Gtk.SeparatorMenuItem()
        menu.append(sep)

        quit_item = Gtk.MenuItem(label="Quit Atlas")
        quit_item.connect("activate", self._on_quit)
        menu.append(quit_item)

        menu.show_all()
        return menu

    def start(self):
        """Start the tray icon in a daemon thread running the GTK main loop."""
        def run_tray():
            self._indicator = AyatanaAppIndicator3.Indicator.new(
                "atlas-voice",
                "AV_ON",
                AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS
            )
            self._indicator.set_icon_theme_path(self._icon_dir)
            self._indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
            self._indicator.set_menu(self._build_menu())
            Gtk.main()

        thread = threading.Thread(target=run_tray, daemon=True)
        thread.start()

    def stop(self):
        """Stop the GTK main loop."""
        GLib.idle_add(Gtk.main_quit)


# ============================================================================
# Main Atlas Class
# ============================================================================

class Atlas:
    def __init__(self):
        self.running = True
        self.wake_model = None
        self.whisper_model = None
        self.audio_buffer = None
        self.tray = None
        self.models_loaded = False

        print("=" * 60)
        print("ATLAS - Always-listening Voice Dictation")
        print("=" * 60)
        print()

        # Initialize audio buffer (starts capturing immediately)
        print("Starting audio capture...")
        self.audio_buffer = AudioBuffer()
        self.audio_buffer.start()
        print("  Audio buffer active.")

        # Load models
        self.load_models()

        # Start system tray icon
        if TRAY_ENABLED:
            print("Starting system tray icon...")
            self.tray = TrayIcon(self)
            self.tray.start()
            print("  Tray icon active. Left-click to pause/resume, right-click for menu.")

        print()
        print('Say "Hey Atlas" to start dictating.')
        print("Press Ctrl+C to quit.")
        print()

    def load_models(self):
        """Load wake word and Whisper models into GPU."""
        if self.models_loaded:
            print("  Models already loaded.")
            return

        # Load wake word model
        print(f"Loading wake word model ({WAKE_WORD_MODEL})...")
        self.wake_model = WakeWordModel(
            wakeword_models=[WAKE_WORD_MODEL],
            inference_framework="tflite"
        )
        print("  Wake word model loaded.")

        # Load Whisper model
        print(f"Loading Whisper model ({WHISPER_MODEL})...")
        self.whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        print("  Whisper model loaded.")

        self.models_loaded = True

    def unload_models(self):
        """Unload models and release GPU resources."""
        if DEBUG_MODE:
            import traceback
            print(f"\n[DEBUG] unload_models() called from:")
            traceback.print_stack(limit=5)

        if not self.models_loaded:
            print("  Models already unloaded.")
            return

        print("Unloading models...")

        # Delete model references to release GPU memory
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
            print("  Whisper model unloaded.")

        if self.wake_model is not None:
            del self.wake_model
            self.wake_model = None
            print("  Wake word model unloaded.")

        # Force garbage collection to release GPU memory
        import gc
        gc.collect()

        # If using CUDA, clear the cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("  CUDA cache cleared.")
        except ImportError:
            pass

        self.models_loaded = False
        print("  GPU resources released.")

    def stop_audio(self):
        """Stop audio capture."""
        if self.audio_buffer:
            print("Stopping audio capture...")
            self.audio_buffer.stop()
            print("  Audio stopped.")

    def restart_audio(self):
        """Restart audio capture with a fresh stream."""
        if self.audio_buffer:
            print("Restarting audio capture...")
            self.audio_buffer.stop()
            time.sleep(0.1)  # Brief pause for device release
            self.audio_buffer.start()
            print("  Audio buffer restarted.")

    def transcribe(self, audio_path):
        """Transcribe audio file with Whisper."""
        segments, info = self.whisper_model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            language="en"
        )
        text = " ".join(segment.text.strip() for segment in segments)
        return process_text(text)

    def transcribe_audio(self, audio_data):
        """Transcribe audio data (int16 numpy array) with Whisper."""
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()

        self.audio_buffer.save_to_wav(audio_data, temp_file.name)

        try:
            text = self.transcribe(temp_file.name)
        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

        return text

    def type_text(self, text):
        """Type text using xdotool."""
        if not AUTO_TYPE or not text:
            return
        subprocess.run(["xdotool", "type", "--clearmodifiers", text])

    def press_enter(self):
        """Press Enter key using xdotool."""
        subprocess.run(["xdotool", "key", "Return"])

    def copy_to_clipboard(self, text):
        """Copy text to clipboard."""
        if not text:
            return
        process = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE
        )
        process.communicate(input=text.encode())

    def listen_for_wake_word(self):
        """
        Listen for wake word using the continuous audio buffer.
        Processes chunks SEQUENTIALLY - wake word model needs every chunk in order.
        Returns timestamp when wake word is detected, or False if paused/stopped.
        """
        # Flush any stale audio from previous session to prevent false triggers
        self.audio_buffer.flush_chunks()

        loop_count = 0
        last_callback_count = self.audio_buffer.callback_count
        chunks_processed = 0

        if DEBUG_MODE:
            print("  [DEBUG] Wake word listener started")

        while self.running:
            # Check if paused or disabled via tray
            if self.tray and not self.tray.is_listening():
                return False

            loop_count += 1

            # Process ALL available chunks (wake word model needs sequential feeding)
            prediction = None
            while True:
                chunk = self.audio_buffer.get_next_chunk()
                if chunk is None:
                    break
                prediction = self.wake_model.predict(chunk)
                chunks_processed += 1

                # Check for wake word detection after each chunk
                for model_name, scores in self.wake_model.prediction_buffer.items():
                    current_score = scores[-1] if len(scores) > 0 else 0
                    if current_score > WAKE_WORD_THRESHOLD:
                        if DEBUG_MODE:
                            print(f"  [DEBUG] Wake word triggered! model={model_name}, score={current_score:.4f}")
                        return time.time()  # Return detection timestamp
                    # Show when score is rising (potential detection)
                    elif DEBUG_MODE and current_score > 0.1:
                        print(f"  [DEBUG] Score rising: {current_score:.4f}")

            # Debug output every ~200ms
            if DEBUG_MODE and loop_count % 20 == 0:
                amp = self.audio_buffer.get_amplitude()
                score = 0
                if prediction:
                    score = list(prediction.values())[0]
                callbacks_delta = self.audio_buffer.callback_count - last_callback_count
                last_callback_count = self.audio_buffer.callback_count

                buffer_len = len(self.audio_buffer.buffer)
                queue_len = self.audio_buffer.chunks_available()
                print(f"  [DEBUG] cb={self.audio_buffer.callback_count} (+{callbacks_delta}), proc={chunks_processed}, q={queue_len}, amp={amp:.0f}, score={score:.4f}")

            time.sleep(0.01)  # Faster polling to keep up with audio

        return False

    def capture_until_silence(self, start_from=None, min_start_time=None):
        """
        Capture audio from the buffer until silence is detected.

        This uses the continuous buffer, so audio is never lost.
        If start_from is provided, backdate capture to that timestamp
        (for catching speech that started during transcription).
        If min_start_time is provided, never backdate before that point
        (prevents capturing wake word audio).
        Returns the captured audio as an int16 numpy array.
        """
        # If we already know speech started, skip the waiting phase
        if start_from:
            print("  Listening... (already have speech, waiting for silence)")
            start_time = start_from
            speech_detected = True
        else:
            print("  Listening... (waiting for speech)")
            start_time = time.time()
            speech_detected = False
        silence_start = None

        while self.running:
            amplitude = self.audio_buffer.get_amplitude()

            if amplitude >= SILENCE_THRESHOLD:
                # Speech detected
                if not speech_detected:
                    speech_detected = True
                    # Backdate start_time to capture any speech that started just before
                    buffer_start = self.audio_buffer.get_speech_start_time(lookback_seconds=1.0)
                    if buffer_start:
                        # Don't backdate before min_start_time (e.g., wake word detection)
                        if min_start_time and buffer_start < min_start_time:
                            buffer_start = min_start_time
                        start_time = buffer_start
                silence_start = None
            elif speech_detected:
                # Silence after speech
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    # Enough silence, stop capturing
                    break

            # Safety limit
            if time.time() - start_time > MAX_RECORD_DURATION:
                print("  (max duration reached)")
                break

            time.sleep(0.05)

        # Extract audio from buffer
        audio_data = self.audio_buffer.get_audio_since(start_time)
        return audio_data

    def run(self):
        """Main loop."""
        last_status_time = 0
        last_callback_count = 0
        STATUS_INTERVAL = 60  # Log status every 60 seconds (reduced for debugging)

        while self.running:
            # Periodic status logging (only when DEBUG_MODE is enabled)
            if DEBUG_MODE:
                now = time.time()
                if now - last_status_time >= STATUS_INTERVAL and self.audio_buffer:
                    last_status_time = now
                    cb_count = self.audio_buffer.callback_count
                    cb_delta = cb_count - last_callback_count
                    last_callback_count = cb_count
                    queue_len = self.audio_buffer.chunks_available()
                    amp = self.audio_buffer.get_amplitude()
                    stream_ok = self.audio_buffer.stream is not None and self.audio_buffer.stream.active
                    state = self.tray.state if self.tray else "NO_TRAY"

                    # Check model state
                    wake_ok = self.wake_model is not None
                    whisper_ok = self.whisper_model is not None
                    models_flag = self.models_loaded

                    print(f"\n[Status] state={state}, models_loaded={models_flag}, wake_model={wake_ok}, whisper_model={whisper_ok}, audio_ok={stream_ok}, cb_delta={cb_delta}, queue={queue_len}, amp={amp:.0f}")

                    # Sanity check: if tray says ON but models are gone, that's a bug
                    if state == TrayIcon.STATE_ON and (not wake_ok or not whisper_ok):
                        print(f"[ERROR] State mismatch! Tray says ON but models missing. wake={wake_ok}, whisper={whisper_ok}")

                    # Sanity check: if audio callbacks stopped, stream is dead
                    if cb_delta == 0 and state == TrayIcon.STATE_ON:
                        print(f"[ERROR] Audio stream dead! No callbacks in {STATUS_INTERVAL}s")

            # Check if disabled (models unloaded)
            if self.tray and self.tray.state == TrayIcon.STATE_DISABLE:
                print("Atlas disabled (GPU released)... use tray menu to enable", end="\r")
                sys.stdout.flush()
                time.sleep(0.5)
                continue

            # Check if paused (OFF state)
            if self.tray and self.tray.state == TrayIcon.STATE_OFF:
                print("Atlas paused... (click tray icon to resume)           ", end="\r")
                sys.stdout.flush()
                time.sleep(0.5)
                continue

            print("Waiting for wake word... (listening)   ", end="\r")
            sys.stdout.flush()

            wake_time = self.listen_for_wake_word()
            if wake_time:
                print(" " * 50, end="\r")
                print("Wake word detected! (say 'break' to finish)")
                play_beep()

                # Update tray to RECORDING state
                if self.tray:
                    self.tray.set_state(TrayIcon.STATE_RECORDING)

                # Track when transcription ends so we can check for missed speech
                last_transcription_end = None
                first_capture = True  # First capture after wake word

                # Continuous dictation loop
                while self.running:
                    # Check if speech started during last transcription
                    capture_from = None
                    if last_transcription_end:
                        speech_start = self.audio_buffer.get_speech_start_time(
                            lookback_seconds=time.time() - last_transcription_end + 1.0,
                            threshold=SILENCE_THRESHOLD
                        )
                        if speech_start and speech_start < time.time() - 0.1:
                            capture_from = speech_start
                            print("  (speech during transcription - backdating capture)")

                    # Capture audio until silence
                    # On first capture, don't backdate before wake word detection
                    min_time = wake_time if first_capture else None
                    first_capture = False
                    audio_data = self.capture_until_silence(start_from=capture_from, min_start_time=min_time)

                    if len(audio_data) == 0:
                        print("  (no audio captured)")
                        continue

                    # Timing info
                    audio_duration = len(audio_data) / SAMPLE_RATE
                    transcription_start = time.time()

                    print(f"  Transcribing ({audio_duration:.1f}s of audio)...")
                    text = self.transcribe_audio(audio_data)

                    last_transcription_end = time.time()
                    transcription_duration = last_transcription_end - transcription_start
                    print(f"  Transcription took {transcription_duration:.1f}s")

                    if not text:
                        print("  (silence - still listening...)")
                        continue

                    # Check for break phrase (allow punctuation between words)
                    # Matches: "break break", "brick break", "Break, brick.", etc.
                    break_pattern = r'\bbr[ei][ae]k\b'
                    if re.search(break_pattern, text, re.IGNORECASE):
                        # Remove break phrase from output
                        text = re.sub(break_pattern + r'[,.\s]*', '', text, flags=re.IGNORECASE).strip()
                        if text:
                            print(f"  Result: {text}")
                            self.copy_to_clipboard(text)
                            self.type_text(text)
                        print("  [SESSION END - pressing Enter]")
                        play_beep()
                        self.press_enter()
                        break
                    else:
                        print(f"  Result: {text}")
                        print()
                        self.copy_to_clipboard(text)
                        self.type_text(text + " ")

                print()

                # Reset wake word model and flush audio buffer to prevent re-trigger
                self.wake_model.reset()
                self.audio_buffer.flush_chunks()

                # Update tray back to ON state
                if self.tray:
                    self.tray.set_state(TrayIcon.STATE_ON)

                time.sleep(0.5)

    def stop(self):
        """Clean shutdown."""
        print("\nExiting...")
        self.running = False
        if self.tray:
            self.tray.stop()
        if self.audio_buffer:
            self.audio_buffer.stop()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    atlas = Atlas()

    def handle_signal(sig, frame):
        print(f"\n[Signal] Received {signal.Signals(sig).name}, shutting down...")
        atlas.stop()
        # Force exit after 3 seconds if graceful shutdown fails
        def force_exit():
            time.sleep(3)
            print("[Signal] Graceful shutdown timed out, forcing exit")
            os._exit(1)
        threading.Thread(target=force_exit, daemon=True).start()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        atlas.run()
    except KeyboardInterrupt:
        atlas.stop()


if __name__ == "__main__":
    main()
