"""Atlas Voice mailbox — thread-safe message passing between GTK and worker threads."""

import threading
from logging_utils import log_debug


class Mailbox:
    # Request constants (toggles, not intents)
    TOGGLE_PAUSE = "toggle_pause"
    TOGGLE_ENABLE = "toggle_enable"
    QUIT = "quit"

    def __init__(self):
        self._pending = None
        self._lock = threading.Lock()
        self._wakeup = threading.Event()  # Event-driven waiting instead of polling

    def post(self, request):
        with self._lock:
            # QUIT is sticky — once posted, cannot be overwritten
            if self._pending != Mailbox.QUIT:
                self._pending = request
                self._wakeup.set()
                log_debug(f"[MAILBOX] Posted: {request}")

    def check(self):
        """Non-blocking check. Returns request or None."""
        with self._lock:
            req = self._pending
            self._pending = None
            self._wakeup.clear()
            if req:
                log_debug(f"[MAILBOX] Read: {req}")
            return req

    def wait(self, timeout):
        """Block until request posted or timeout. Returns request or None."""
        self._wakeup.wait(timeout)
        return self.check()
