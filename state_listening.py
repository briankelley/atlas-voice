"""Atlas Voice state: listening — wake word detection loop.

State transition invariant:
    Any state detecting audio failure with ``device_name`` configured MUST
    return ``"listening"``.  The listening state is the sole recovery
    orchestrator.  Entry to listening from any state must be safe — no
    resources held that require cleanup beyond what the returning state
    handles before its return.
"""

import random
import time

from logging_utils import log_debug, log_info, log_error
from mailbox import Mailbox
from context import handle_quit


def _wait_for_device(ctx, first_boot=False):
    """Poll for device reconnection with exponential backoff.

    Returns: "_recovered" on success, or a state string from mailbox handling.
    """
    device_name = ctx.audio_buffer.device_name
    ctx.set_icon("AV_DISABLE")
    log_info(f"[AUDIO] Device disconnected. Polling for reconnect... (device='{device_name}')")

    # Tear down old stream aggressively (hardware may have vanished)
    ctx.audio_buffer.stop(force=True)
    ctx.audio_buffer.flush_chunk_queue()

    # Backoff parameters
    interval = 2.0
    MAX_INTERVAL = 15.0
    poll_start_time = time.time()
    last_log_time = 0.0
    config_warning_shown = False
    models_unloaded = False
    RESOURCE_RELEASE_SECS = 600  # 10 minutes

    while True:
        # Apply jitter: ±20%
        sleep_time = interval * random.uniform(0.8, 1.2)
        time.sleep(sleep_time)

        # Check mailbox each iteration (responsive to user commands)
        req = ctx.mailbox.check()
        if req == Mailbox.TOGGLE_PAUSE:
            log_info("[STATE] Exiting: device poll -> paused")
            return "paused"
        elif req == Mailbox.TOGGLE_ENABLE:
            ctx.unload_models()
            ctx.audio_buffer.stop(force=True)
            log_info("[STATE] Exiting: device poll -> disabled")
            return "disabled"
        elif req == Mailbox.QUIT:
            log_info("[STATE] Exiting: device poll -> shutdown")
            return handle_quit(ctx)

        elapsed = time.time() - poll_start_time

        # Resource release after 10 minutes of continuous polling
        if not models_unloaded and elapsed >= RESOURCE_RELEASE_SECS:
            ctx.unload_models()
            models_unloaded = True
            log_info("[AUDIO] Models unloaded to free resources during extended disconnect")

        # Config validity warning (first boot only, after 60s with no sighting)
        if first_boot and not config_warning_shown and elapsed >= 60.0:
            log_error(f"[AUDIO] Device '{device_name}' not found — verify settings.conf [audio] device value")
            config_warning_shown = True

        # Rate-limited "still polling" log (once per 30 seconds)
        now = time.time()
        if now - last_log_time >= 30.0:
            log_info(f"[AUDIO] Still polling for device '{device_name}'... ({int(elapsed)}s elapsed)")
            last_log_time = now

        # Check if device is present
        if not ctx.audio_buffer.is_device_present():
            # Exponential backoff
            interval = min(interval * 2, MAX_INTERVAL)
            continue

        # Device found — reset backoff and attempt stream start
        interval = 2.0
        log_info(f"[AUDIO] Device '{device_name}' detected, attempting stream start")

        try:
            ctx.audio_buffer.start()
        except Exception as e:
            log_error(f"[AUDIO] Stream start failed after device found: {e}")
            # Continue polling with backoff
            interval = min(interval * 2, MAX_INTERVAL)
            continue

        # Wait for first audio callback to confirm stream is truly functional
        if not ctx.audio_buffer._first_callback_event.wait(2.0):
            log_error("[AUDIO] Callback not received after stream start, retrying...")
            ctx.audio_buffer.stop(force=True)
            interval = min(interval * 2, MAX_INTERVAL)
            continue

        # Full success — stream is running and receiving audio
        log_info("[AUDIO] Stream recovered successfully")
        ctx.set_icon("AV_ON")
        return "_recovered"


def state_listening(ctx):
    """Listening state. Process audio chunks for wake word detection.

    PRE: ctx.wake_model is loaded (non-None). ctx.audio_buffer may or may not
         be running (handles startup recovery if stream is None).
    POST: On wake detection: sets ctx.wake_time, ctx.recording_mode = "wake".
          Clears ctx.wake_time, ctx.captured_audio, ctx.recording_mode on entry.
    RETURNS: "recording" (wake detected), "paused" (toggle),
             "disabled" (toggle/audio failure), None (quit).
    INTERRUPTS: Mailbox TOGGLE_PAUSE, TOGGLE_ENABLE, QUIT checked every iteration.
    """
    log_info("[STATE] Entering: listening")
    ctx.set_icon("AV_ON")

    # Ensure audio buffer is running (handles initial startup and recovery)
    if ctx.audio_buffer.stream is None:
        try:
            ctx.audio_buffer.start()
        except ValueError as e:
            if ctx.audio_buffer.device_name:
                log_error(f"[AUDIO] Device not found at startup: {e}")
                result = _wait_for_device(ctx, first_boot=True)
                if result == "_recovered":
                    pass  # Continue to listening loop below
                else:
                    return result
            else:
                log_error(f"[AUDIO] Start failed: {e}")
                return "disabled"

    # Clear inter-state data
    ctx.clear_interstate_data()

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
                if ctx.audio_buffer.restart():
                    log_info("[AUDIO] Stream restarted successfully")
                    continue
                else:
                    if ctx.audio_buffer.device_name:
                        result = _wait_for_device(ctx)
                        if result == "_recovered":
                            last_health_check = time.time()
                            ctx.audio_buffer.flush_chunk_queue()
                            if ctx.wake_model is not None:
                                try:
                                    ctx.wake_model.reset()
                                except Exception:
                                    pass
                            log_info('Waiting for wake word... (say "Hey Atlas")')
                            continue
                        else:
                            return result
                    else:
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
