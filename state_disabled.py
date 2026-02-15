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
                ctx.load_models()
                # Start audio stream if not running
                try:
                    ctx.audio_buffer.start()
                except Exception:
                    pass  # May already be started
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
