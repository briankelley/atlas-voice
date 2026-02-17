"""Atlas Voice state: recording — capturing speech until silence."""

import time

import numpy as np

from logging_utils import log_debug, log_info, log_error
from mailbox import Mailbox
from context import handle_quit


def state_recording(ctx):
    """Recording state. Capture audio until silence detected.

    PRE: ctx.recording_mode is "wake" or "vad".
         If "wake": ctx.wake_time is set (non-None).
         ctx.audio_buffer is running.
    POST: Sets ctx.captured_audio with recorded audio data.
    RETURNS: "transcribing" (audio captured), "listening" (no speech/VAD timeout),
             "paused" (toggle), "disabled" (toggle/audio failure), None (quit).
    INTERRUPTS: Mailbox TOGGLE_PAUSE, TOGGLE_ENABLE, QUIT checked every iteration.
    """
    log_info("[STATE] Entering: recording")
    ctx.set_icon("AV_RECORDING")
    ctx.play_beep()

    # Validate inter-state data
    if ctx.recording_mode == "wake" and ctx.wake_time is None:
        log_error("[STATE] Entered recording in wake mode with no wake_time, returning to listening")
        ctx.clear_interstate_data()
        return "listening"

    # Clear output data
    ctx.captured_audio = None

    # Accumulate audio chunks
    audio_chunks = []
    silence_start = None
    recording_start = time.time()

    silence_threshold = ctx.config['silence_threshold']
    silence_duration = ctx.config['silence_duration']
    max_duration = ctx.config['max_record_duration']

    # Audio health check tracking
    last_health_check = time.time()
    HEALTH_CHECK_INTERVAL = 2.0

    # Wake mode: reach back to capture first word
    if ctx.recording_mode == "wake":
        if ctx.audio_buffer.has_audio_since(ctx.wake_time):
            historical = ctx.audio_buffer.get_audio_since(ctx.wake_time)
            if len(historical) > 0:
                audio_chunks.append(historical)
                log_debug(f"[AUDIO] Reached back {len(historical)} samples from wake time")
        else:
            log_error("[AUDIO] Wake time too old, not in ring buffer")

    # VAD mode: wait for speech onset before capturing
    elif ctx.recording_mode == "vad":
        log_debug("[AUDIO] VAD mode: waiting for speech onset")
        vad_timeout = ctx.config['vad_timeout']
        vad_start = time.time()
        speech_found = False

        while time.time() - vad_start < vad_timeout:
            chunk = ctx.audio_buffer.get_chunk(timeout=0.08)
            if chunk is not None and np.abs(chunk).mean() >= silence_threshold:
                speech_found = True
                audio_chunks.append(chunk)
                break

            # Check mailbox during VAD wait
            req = ctx.mailbox.check()
            if req == Mailbox.TOGGLE_PAUSE:
                log_info("[STATE] Recording interrupted during VAD wait")
                return "paused"
            elif req == Mailbox.TOGGLE_ENABLE:
                log_info("[STATE] Recording interrupted during VAD wait")
                ctx.unload_models()
                try:
                    ctx.audio_buffer.stop()
                except Exception:
                    pass
                return "disabled"
            elif req == Mailbox.QUIT:
                return handle_quit(ctx)

        if not speech_found:
            log_debug("[AUDIO] VAD timeout, no speech detected")
            return "listening"

    log_info("  Listening... (capturing speech, waiting for silence)")

    while True:
        chunk = ctx.audio_buffer.get_chunk(timeout=0.08)
        if chunk is not None:
            audio_chunks.append(chunk)

            # Silence detection
            amplitude = np.abs(chunk).mean()
            if amplitude < silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_duration:
                    log_debug(f"[AUDIO] Silence detected after {time.time() - recording_start:.1f}s")
                    break
            else:
                silence_start = None

            # Max duration check
            if time.time() - recording_start > max_duration:
                log_info("[AUDIO] Max recording duration reached")
                break

        # Periodic audio health check
        now = time.time()
        if now - last_health_check >= HEALTH_CHECK_INTERVAL:
            last_health_check = now
            if not ctx.audio_buffer.is_healthy():
                log_error("[AUDIO] Stream unhealthy during recording, attempting restart")
                if not ctx.audio_buffer.restart():
                    if ctx.audio_buffer.device_name:
                        log_info("[STATE] Exiting: recording -> listening (device recovery)")
                        return "listening"
                    else:
                        log_error("[AUDIO] Restart failed during recording")
                        ctx.unload_models()
                        return "disabled"

        # Check mailbox every iteration
        req = ctx.mailbox.check()
        if req == Mailbox.TOGGLE_PAUSE:
            log_info("[STATE] Recording interrupted, discarding audio")
            log_info("[STATE] Exiting: recording -> paused")
            return "paused"
        elif req == Mailbox.TOGGLE_ENABLE:
            log_info("[STATE] Recording interrupted, discarding audio")
            ctx.unload_models()
            try:
                ctx.audio_buffer.stop()
            except Exception:
                pass
            log_info("[STATE] Exiting: recording -> disabled")
            return "disabled"
        elif req == Mailbox.QUIT:
            log_info("[STATE] Exiting: recording -> shutdown")
            return handle_quit(ctx)

    # End of recording
    if len(audio_chunks) == 0:
        log_debug("[AUDIO] No speech captured, returning to listening")
        return "listening"

    # Concatenate all audio chunks
    ctx.captured_audio = np.concatenate(audio_chunks)
    audio_duration = len(ctx.captured_audio) / ctx.config['sample_rate']
    log_info(f"  Captured {audio_duration:.1f}s of audio")

    log_info("[STATE] Exiting: recording -> transcribing")
    return "transcribing"
