"""Atlas Voice state: disabled — models unloaded, GPU free."""

from logging_utils import log_debug, log_info, log_error
from mailbox import Mailbox
from context import handle_quit


def state_disabled(ctx):
    """Disabled state. Models unloaded, GPU free. Wait for TOGGLE_ENABLE.

    PRE: Models are unloaded, GPU is free.
    POST: On enable: loads models, starts audio buffer.
    RETURNS: "listening" (enable success), None (quit).
    INTERRUPTS: Mailbox TOGGLE_ENABLE, QUIT via blocking wait; TOGGLE_PAUSE ignored.
    """
    log_info("[STATE] Entering: disabled")
    ctx.set_icon("AV_DISABLE")
    log_info("Atlas disabled (GPU released). Use tray menu to enable.")

    while True:
        req = ctx.mailbox.wait(timeout=0.5)
        if req == Mailbox.TOGGLE_ENABLE:
            log_info("[STATE] Enabling (loading models)...")
            try:
                # Check device presence before loading models to avoid
                # wasting GPU memory if the device is unavailable
                if ctx.audio_buffer.device_name and not ctx.audio_buffer.is_device_present():
                    log_error(f"[STATE] Audio device '{ctx.audio_buffer.device_name}' not found, staying disabled")
                    continue

                ctx.load_models()

                # Start audio stream if not running
                try:
                    ctx.audio_buffer.start()
                except Exception as e:
                    log_error(f"[STATE] Audio start failed after model load: {e}")
                    # Unload models to free GPU since we can't proceed
                    ctx.unload_models()
                    continue

                log_info("[STATE] Exiting: disabled -> listening")
                return "listening"
            except Exception as e:
                log_error(f"[STATE] Model load failed, staying disabled: {e}")
                continue
        elif req == Mailbox.TOGGLE_PAUSE:
            log_debug("[TRAP] In DISABLED, got TOGGLE_PAUSE (nonsensical, ignored)")
        elif req == Mailbox.QUIT:
            log_info("[STATE] Exiting: disabled -> shutdown")
            return handle_quit(ctx)
