"""Atlas Voice state: paused — models loaded, not listening."""

from logging_utils import log_debug, log_info, log_error
from mailbox import Mailbox
from context import handle_quit


def state_paused(ctx):
    """Paused state. Models loaded but not listening for wake word.

    PRE: Models are loaded (wake + whisper). Audio buffer may be running.
    POST: No state changes.
    RETURNS: "listening" (toggle pause), "disabled" (toggle enable), None (quit).
    INTERRUPTS: Mailbox TOGGLE_PAUSE, TOGGLE_ENABLE, QUIT via blocking wait.
    """
    log_info("[STATE] Entering: paused")
    ctx.set_icon("AV_OFF")

    while True:
        req = ctx.mailbox.wait(timeout=0.2)
        if req == Mailbox.TOGGLE_PAUSE:
            log_info("[STATE] Exiting: paused -> listening")
            return "listening"
        elif req == Mailbox.TOGGLE_ENABLE:
            log_info("[STATE] Exiting: paused -> disabled (unloading models)")
            ctx.unload_models()
            try:
                ctx.audio_buffer.stop()
            except Exception:
                pass
            return "disabled"
        elif req == Mailbox.QUIT:
            log_info("[STATE] Exiting: paused -> shutdown")
            return handle_quit(ctx)
