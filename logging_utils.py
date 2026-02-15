"""Atlas Voice logging utilities."""

from datetime import datetime

# Module-level flags (set by main.py after config load)
_DEBUG = False
_LOG_TRANSCRIPTS = False


def set_debug(value):
    global _DEBUG
    _DEBUG = value


def set_log_transcripts(value):
    global _LOG_TRANSCRIPTS
    _LOG_TRANSCRIPTS = value


def log_debug(msg):
    """Debug messages (only when debug=true)."""
    if _DEBUG:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {msg}", flush=True)


def log_info(msg):
    """Info messages (always printed)."""
    print(msg, flush=True)


def log_error(msg):
    """Error messages (always printed)."""
    print(f"[ERROR] {msg}", flush=True)


def should_log_transcripts():
    """Check if transcription content should be logged."""
    return _LOG_TRANSCRIPTS
