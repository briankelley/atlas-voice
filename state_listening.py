"""Atlas Voice state: listening — wake word detection loop."""

import time

from logging_utils import log_debug, log_info, log_error
from mailbox import Mailbox
from context import handle_quit


def state_listening(ctx):
    """Listening state. Process audio chunks for wake word detection.

    PRE: ctx.wake_model is loaded (non-None). ctx.audio_buffer is running.
    POST: On wake detection: sets ctx.wake_time, ctx.recording_mode = "wake".
          Clears ctx.wake_time, ctx.captured_audio, ctx.recording_mode on entry.
    RETURNS: "recording" (wake detected), "paused" (toggle),
             "disabled" (toggle/audio failure), None (quit).
    INTERRUPTS: Mailbox TOGGLE_PAUSE, TOGGLE_ENABLE, QUIT checked every iteration.
    """
    log_info("[STATE] Entering: listening")
    ctx.set_icon("AV_ON")

    # Clear inter-state data
    ctx.wake_time = None
    ctx.captured_audio = None
    ctx.recording_mode = None

    # Flush wake word processing queue (NOT the ring buffer)
    ctx.audio_buffer.flush_chunk_queue()

    # Reset wake model prediction buffer to prevent false triggers
    if ctx.wake_model is not None:
        try:
            ctx.wake_model.reset()
        except Exception as e:
            log_error(f"[WAKE] Model reset failed: {e}")

    # Health check tracking
    last_health_check = time.time()
    HEALTH_CHECK_INTERVAL = 2.0

    log_info('Waiting for wake word... (say "Hey Atlas")')

    while True:
        # Periodic health check
        now = time.time()
        if now - last_health_check >= HEALTH_CHECK_INTERVAL:
            last_health_check = now
            if not ctx.audio_buffer.is_healthy():
                log_error("[AUDIO] Stream unhealthy, attempting restart")
                try:
                    ctx.audio_buffer.restart()
                except Exception as e:
                    log_error(f"[AUDIO] Restart failed: {e}")
                    ctx.unload_models()
                    log_info("[STATE] Exiting: listening -> disabled (audio failure)")
                    return "disabled"

        # Process audio chunks for wake word
        chunk = ctx.audio_buffer.get_chunk(timeout=0.08)
        if chunk is not None:
            try:
                ctx.wake_model.predict(chunk)

                # Check prediction buffer for wake word detection
                for model_name, scores in ctx.wake_model.prediction_buffer.items():
                    if len(scores) > 0 and scores[-1] > ctx.config['wake_word_threshold']:
                        log_info(f"[WAKE] Detected! model={model_name}, score={scores[-1]:.4f}")
                        ctx.wake_time = time.time()
                        ctx.recording_mode = "wake"
                        log_info("[STATE] Exiting: listening -> recording")
                        return "recording"
                    elif ctx.debug and len(scores) > 0 and scores[-1] > 0.1:
                        log_debug(f"[WAKE] Score rising: {scores[-1]:.4f}")
            except Exception as e:
                log_error(f"[WAKE] Prediction failed: {e}")

        # Check mailbox every iteration
        req = ctx.mailbox.check()
        if req == Mailbox.TOGGLE_PAUSE:
            log_info("[STATE] Exiting: listening -> paused")
            return "paused"
        elif req == Mailbox.TOGGLE_ENABLE:
            log_info("[STATE] Exiting: listening -> disabled (unloading models)")
            ctx.unload_models()
            try:
                ctx.audio_buffer.stop()
            except Exception:
                pass
            return "disabled"
        elif req == Mailbox.QUIT:
            log_info("[STATE] Exiting: listening -> shutdown")
            return handle_quit(ctx)
