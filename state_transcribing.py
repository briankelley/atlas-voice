"""Atlas Voice state: transcribing — Whisper inference and text output."""

import os
import time
import tempfile
import re

import numpy as np

from logging_utils import log_debug, log_info, log_error, should_log_transcripts
from mailbox import Mailbox
from context import handle_quit
from text_processing import process_text, contains_break_keyword, remove_break_keyword
from text_output import output_text, copy_to_clipboard, press_enter


def state_transcribing(ctx):
    """Transcribing state. Run Whisper inference, output text, handle continuous dictation.

    PRE: ctx.captured_audio is set (non-None, non-empty).
         ctx.whisper_model is loaded.
         ctx.audio_buffer is running.
    POST: Clears ctx.captured_audio. Outputs transcribed text.
          May set ctx.recording_mode and ctx.wake_time for continuous dictation.
    RETURNS: "recording" (continuous dictation), "listening" (session end/empty),
             "paused" (toggle), "disabled" (toggle/audio failure), None (quit).
    INTERRUPTS: Mailbox TOGGLE_PAUSE, TOGGLE_ENABLE, QUIT checked pre- and post-transcription.
    """
    log_info("[STATE] Entering: transcribing")
    ctx.set_icon("AV_RECORDING")  # Same icon as recording

    # Validate
    if ctx.captured_audio is None or len(ctx.captured_audio) == 0:
        log_error("[STATE] Entered transcribing with no audio, returning to listening")
        return "listening"

    # Pre-transcription mailbox check
    req = ctx.mailbox.check()
    if req == Mailbox.QUIT:
        return handle_quit(ctx)
    elif req == Mailbox.TOGGLE_PAUSE:
        log_info("[STATE] Pre-transcription pause, discarding audio")
        ctx.captured_audio = None
        return "paused"
    elif req == Mailbox.TOGGLE_ENABLE:
        log_info("[STATE] Pre-transcription disable, discarding audio")
        ctx.captured_audio = None
        ctx.unload_models()
        try:
            ctx.audio_buffer.stop()
        except Exception:
            pass
        return "disabled"

    # Create secure temp file
    temp_dir = os.environ.get('XDG_RUNTIME_DIR', tempfile.gettempdir())
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        prefix="atlas_voice_",
        dir=temp_dir,
        suffix=".wav"
    )
    temp_path = temp_file.name
    temp_file.close()

    # Audio health check before transcription
    if not ctx.audio_buffer.is_healthy():
        log_error("[AUDIO] Stream unhealthy before transcription, attempting restart")
        try:
            ctx.audio_buffer.restart()
        except Exception as e:
            log_error(f"[AUDIO] Restart failed: {e}")
            ctx.captured_audio = None
            ctx.unload_models()
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            return "disabled"

    text = ""
    try:
        # Save audio to temp WAV
        ctx.audio_buffer.save_to_wav(ctx.captured_audio, temp_path)

        # Transcribe (blocking)
        audio_duration = len(ctx.captured_audio) / ctx.config['sample_rate']
        log_info(f"  Transcribing ({audio_duration:.1f}s of audio)...")
        start_time = time.time()

        try:
            segments, info = ctx.whisper_model.transcribe(
                temp_path,
                beam_size=5,
                vad_filter=True,
                language="en"
            )
            text = " ".join(segment.text.strip() for segment in segments)
            duration = time.time() - start_time

            if should_log_transcripts():
                log_info(f'[TRANSCRIBE] Complete in {duration:.1f}s: "{text}"')
            else:
                log_info(f"[TRANSCRIBE] Complete in {duration:.1f}s, {len(text)} chars")
        except Exception as e:
            log_error(f"[TRANSCRIBE] Failed: {e}")
            ctx.captured_audio = None
            return "listening"

    finally:
        # Always delete temp file
        try:
            os.unlink(temp_path)
        except Exception as e:
            log_error(f"[TEMP] Failed to delete {temp_path}: {e}")

    # Clear consumed data
    transcription_end = time.time()
    ctx.captured_audio = None

    # Audio health check after transcription
    if not ctx.audio_buffer.is_healthy():
        log_error("[AUDIO] Stream unhealthy after transcription, attempting restart")
        try:
            ctx.audio_buffer.restart()
        except Exception as e:
            log_error(f"[AUDIO] Restart failed: {e}")
            ctx.unload_models()
            return "disabled"

    # Handle empty transcription
    if not text or not text.strip():
        log_info("  (silence - no speech detected)")
        return "listening"

    # Post-transcription mailbox check
    req = ctx.mailbox.check()
    if req == Mailbox.QUIT:
        # Work is done, deliver it before quitting
        processed = process_text(text, ctx.config)
        if processed:
            output_text(processed, ctx)
        return handle_quit(ctx)
    elif req == Mailbox.TOGGLE_PAUSE:
        processed = process_text(text, ctx.config)
        if processed:
            output_text(processed, ctx)
        return "paused"
    elif req == Mailbox.TOGGLE_ENABLE:
        processed = process_text(text, ctx.config)
        if processed:
            output_text(processed, ctx)
        ctx.unload_models()
        try:
            ctx.audio_buffer.stop()
        except Exception:
            pass
        return "disabled"

    # Check for mode switch commands
    text = _handle_mode_switch(text, ctx)

    # Process text through punctuation/word replacement pipeline
    text = process_text(text, ctx.config)

    if not text:
        log_info("  (empty after processing)")
        return "listening"

    # Check for break keyword
    if contains_break_keyword(text, ctx.config):
        # Remove break keyword, output remaining text
        remaining = remove_break_keyword(text, ctx.config)
        if remaining:
            log_info(f"  Result: {remaining}")
            copy_to_clipboard(remaining)
            output_text(remaining, ctx)
        log_info("  [SESSION END - pressing Enter]")
        ctx.play_beep()
        press_enter()
        log_info("[STATE] Exiting: transcribing -> listening")
        return "listening"

    # Normal output
    log_info(f"  Result: {text}")
    copy_to_clipboard(text)
    output_text(text + " ", ctx)  # Trailing space for continuous dictation

    # Check for overlapping speech (continuous dictation)
    overlap_start = ctx.audio_buffer.detect_speech_during(start_time, transcription_end)
    if overlap_start is not None:
        log_debug("[SESSION] Overlap detected, continuing dictation")
        ctx.wake_time = overlap_start
        ctx.recording_mode = "wake"
        log_info("[STATE] Exiting: transcribing -> recording (overlap)")
        return "recording"
    else:
        # No overlap: continue dictation session with VAD mode
        log_debug("[SESSION] No overlap, waiting for next utterance (VAD mode)")
        ctx.recording_mode = "vad"
        log_info("[STATE] Exiting: transcribing -> recording (VAD)")
        return "recording"


def _handle_mode_switch(text, ctx):
    """Check for and handle mode switch voice commands. Returns cleaned text."""
    console_phrase = ctx.config.get('switch_to_console_phrase', '')
    gui_phrase = ctx.config.get('switch_to_gui_phrase', '')

    if console_phrase:
        pattern = re.compile(re.escape(console_phrase) + r'[,.\s]*', re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub('', text).strip()
            ctx.typing_mode = 'console'
            log_info("  [MODE SWITCHED TO CONSOLE]")

    if gui_phrase:
        pattern = re.compile(re.escape(gui_phrase) + r'[,.\s]*', re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub('', text).strip()
            ctx.typing_mode = 'gui'
            log_info("  [MODE SWITCHED TO GUI]")

    return text
