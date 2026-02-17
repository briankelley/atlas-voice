"""Atlas Voice context — shared state for all state functions."""

import os
import subprocess
import gc

from logging_utils import log_debug, log_info, log_error

# Lazy imports for models (heavy dependencies)
# These are imported inside load_models() to avoid import-time GPU init

try:
    from gi.repository import GLib
except ImportError:
    GLib = None


class AtlasContext:
    def __init__(self, config, audio_buffer):
        self.config = config
        self.audio_buffer = audio_buffer
        self.tray = None          # set after TrayIcon created
        self.wake_model = None    # loaded/unloaded by state functions
        self.whisper_model = None # loaded/unloaded by state functions

        # Mailbox (created separately, passed in or set after)
        self.mailbox = None

        # Inter-state data passing
        self.wake_time = None       # set by state_listening, read by state_recording
        self.captured_audio = None  # set by state_recording, read by state_transcribing
        self.recording_mode = None  # "wake" or "vad"

        # Flags
        self.debug = config.get('debug', False)
        self.log_transcripts = config.get('log_transcripts', False)
        self.typing_mode = config.get('typing_mode', 'console')
        self.tray_running = False
        self._pending_icon = None   # Last icon requested (for startup race)

    def clear_interstate_data(self):
        """Reset all inter-state data fields to None.

        Call at state entry and on error-return paths to prevent stale data.
        """
        self.wake_time = None
        self.captured_audio = None
        self.recording_mode = None

    def set_icon(self, icon_name):
        """Tell GTK thread to display this icon. Called from worker thread."""
        log_debug(f"[ICON] Setting: {icon_name}")
        self._pending_icon = icon_name
        if self.tray and self.tray_running:
            try:
                GLib.idle_add(self.tray.set_icon_by_name, icon_name)
            except Exception as e:
                log_error(f"[ICON] Failed to set icon: {e}")

    def load_models(self):
        """Load wake word + whisper models onto GPU. Atomic: if any fails, all unloaded."""
        from openwakeword.model import Model as WakeWordModel
        from faster_whisper import WhisperModel

        try:
            log_info("[MODELS] Loading wake word model...")
            self.wake_model = WakeWordModel(
                wakeword_models=[self.config['wake_word_model']],
                inference_framework="tflite"
            )
            log_info("[MODELS] Wake word model loaded")

            log_info("[MODELS] Loading Whisper model...")
            self.whisper_model = WhisperModel(
                self.config['whisper_model'],
                device=self.config['whisper_device'],
                compute_type=self.config['whisper_compute_type']
            )
            log_info("[MODELS] Whisper model loaded")
            log_info("[MODELS] Models loaded")
        except Exception as e:
            log_error(f"[MODELS] Load failed: {e}")
            self.unload_models()
            raise

    def unload_models(self):
        """Unload models, free GPU memory. Idempotent, no-throw."""
        try:
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
                log_debug("[MODELS] Whisper unloaded")
        except Exception as e:
            log_error(f"[MODELS] Whisper unload failed: {e}")

        try:
            if self.wake_model is not None:
                del self.wake_model
                self.wake_model = None
                log_debug("[MODELS] Wake model unloaded")
        except Exception as e:
            log_error(f"[MODELS] Wake model unload failed: {e}")

        # Force garbage collection and clear CUDA cache
        try:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception:
            pass

        log_info("[MODELS] GPU freed")

    def play_beep(self):
        """Play notification sound (non-blocking subprocess)."""
        if not self.config.get('beep_on_wake', True):
            return
        beep_sound = self.config.get('beep_sound')
        if beep_sound and os.path.exists(beep_sound):
            try:
                subprocess.Popen(
                    ['paplay', beep_sound],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                log_error(f"[AUDIO] Beep failed: {e}")


def handle_quit(ctx):
    """Common cleanup for QUIT. Returns None (shutdown state)."""
    log_info("[SHUTDOWN] QUIT received")
    try:
        ctx.unload_models()
    except Exception as e:
        log_error(f"[SHUTDOWN] Model unload failed: {e}")

    try:
        ctx.audio_buffer.stop()
    except Exception as e:
        log_error(f"[SHUTDOWN] Audio stop failed: {e}")

    try:
        ctx.tray_running = False
        if ctx.tray:
            ctx.tray.stop()
    except Exception as e:
        log_error(f"[SHUTDOWN] Tray stop failed: {e}")

    return None  # Exit state loop
