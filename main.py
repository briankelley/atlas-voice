#!/usr/bin/env python3
"""Atlas Voice — Always-listening voice dictation with wake word detection."""

import os
import sys

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

import signal
import tempfile
import threading
import traceback

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import logging_utils
from logging_utils import log_info, log_error, log_debug
from config import load_config
from mailbox import Mailbox
from audio_buffer import AudioBuffer
from context import AtlasContext, handle_quit
from tray import TrayIcon

# State function imports
from state_disabled import state_disabled
from state_paused import state_paused
from state_listening import state_listening
from state_recording import state_recording
from state_transcribing import state_transcribing


# State dispatch table
STATES = {
    "disabled": state_disabled,
    "paused": state_paused,
    "listening": state_listening,
    "recording": state_recording,
    "transcribing": state_transcribing,
}


def state_worker(ctx):
    """Worker thread: runs state machine loop."""
    state = "disabled"
    while state is not None:
        if state not in STATES:
            log_error(f"Invalid state '{state}' returned, shutting down")
            state = None
        else:
            log_debug(f"[STATE] Dispatching: {state}")
            state = STATES[state](ctx)
    log_info("State worker exiting")


def cleanup_orphaned_temp_files():
    """Delete orphaned atlas temp files from previous crashes."""
    temp_dir = os.environ.get('XDG_RUNTIME_DIR', tempfile.gettempdir())
    try:
        for filename in os.listdir(temp_dir):
            if filename.startswith('atlas_voice_') and filename.endswith('.wav'):
                path = os.path.join(temp_dir, filename)
                try:
                    os.unlink(path)
                    log_debug(f"[TEMP] Cleaned up orphaned file: {filename}")
                except Exception as e:
                    log_error(f"[TEMP] Failed to clean {filename}: {e}")
    except Exception as e:
        log_error(f"[TEMP] Orphaned file cleanup failed: {e}")


def main():
    try:
        print("=" * 60)
        print("ATLAS - Always-listening Voice Dictation")
        print("=" * 60)
        print()

        # Load config
        config = load_config()
        logging_utils.set_debug(config.get('debug', False))
        logging_utils.set_log_transcripts(config.get('log_transcripts', False))

        # Cleanup orphaned temp files from previous crashes
        cleanup_orphaned_temp_files()

        # Create mailbox
        mailbox = Mailbox()

        # Create audio buffer
        audio_buffer = AudioBuffer(config)

        # Create context
        ctx = AtlasContext(config, audio_buffer)
        ctx.mailbox = mailbox

        # Register signal handlers (after ctx created)
        def handle_signal(sig, frame):
            log_info(f"Signal {sig} received")
            ctx.mailbox.post(Mailbox.QUIT)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        # Create tray (on main thread, before GTK loop)
        tray = TrayIcon(mailbox, config['icon_dir'])
        tray.setup()
        ctx.tray = tray
        ctx.tray_running = True

        # Apply pending icon if any (for startup race)
        if ctx._pending_icon:
            tray.set_icon_by_name(ctx._pending_icon)

        # Start worker thread (state machine)
        worker = threading.Thread(target=state_worker, args=(ctx,), daemon=False)
        worker.start()

        # Run GTK main loop (on main thread)
        log_info("Atlas Voice started")
        print()
        print("System tray icon active. Right-click for menu.")
        print("Press Ctrl+C to quit.")
        print()
        Gtk.main()

        # GTK exited, wait for worker
        worker.join(timeout=2.0)
        if worker.is_alive():
            log_error("Worker thread did not exit cleanly")

    except Exception as e:
        log_error(f"Unhandled exception: {e}")
        traceback.print_exc()

    finally:
        # Final cleanup — guaranteed to run, each step individually guarded
        log_info("[SHUTDOWN] Final cleanup")

        if 'ctx' in locals():
            try:
                ctx.unload_models()
            except Exception as e:
                log_error(f"[SHUTDOWN] Model unload failed: {e}")

            try:
                ctx.audio_buffer.stop()
            except Exception as e:
                log_error(f"[SHUTDOWN] Audio stop failed: {e}")

            try:
                if ctx.tray:
                    ctx.tray_running = False
            except Exception as e:
                log_error(f"[SHUTDOWN] Tray cleanup failed: {e}")

        log_info("Atlas Voice stopped.")


if __name__ == '__main__':
    main()
