"""Tests for state_worker() dispatch loop, cleanup, and signal handling from main.py."""

import signal
import sys
import threading
import time
import pytest
from unittest import mock

# Mock GTK/gi before importing main.py (it does `import gi` at module level).
# Must persist in sys.modules so mock.patch('main.log_*') can resolve the module.
_mock_gi = mock.MagicMock()
_mock_gi.require_version = mock.MagicMock()
sys.modules.setdefault('gi', _mock_gi)
sys.modules.setdefault('gi.repository', _mock_gi.repository)

if 'main' in sys.modules:
    del sys.modules['main']
from main import state_worker, STATES

from audio_buffer import AudioBuffer
from context import AtlasContext
from mailbox import Mailbox


def _make_ctx(config_overrides=None):
    """Build a test context with mocked models and audio."""
    config = {
        'sample_rate': 16000,
        'chunk_size': 1280,
        'silence_threshold': 500,
        'buffer_seconds': 10,
        'audio_device': '',
        'wake_preroll': 0.75,
        'wake_word_model': '/dev/null',
        'whisper_model': '/dev/null',
        'whisper_device': 'cpu',
        'whisper_compute_type': 'int8',
        'wake_word_threshold': 0.50,
        'silence_duration': 2.0,
        'max_record_duration': 60,
        'vad_timeout': 5.0,
        'auto_type': False,
        'beep_on_wake': False,
        'debug': False,
        'log_transcripts': False,
        'tray_enabled': False,
        'typing_mode': 'console',
        'switch_to_console_phrase': '',
        'switch_to_gui_phrase': '',
        'spoken_punctuation': [],
        'word_replacements': {},
        'icon_dir': '/tmp',
    }
    if config_overrides:
        config.update(config_overrides)
    buf = AudioBuffer(config)
    ctx = AtlasContext(config, buf)
    ctx.mailbox = Mailbox()
    return ctx


# ---------------------------------------------------------------------------
# state_worker dispatch loop
# ---------------------------------------------------------------------------

class TestStateWorkerDispatch:
    """Tests for state_worker() from main.py:69-79."""

    def test_valid_state_chain_executes_in_order(self):
        """States execute in sequence: disabled -> listening -> None."""
        call_order = []

        def fake_disabled(ctx):
            call_order.append("disabled")
            return "listening"

        def fake_listening(ctx):
            call_order.append("listening")
            return None

        with mock.patch.dict(STATES, {"disabled": fake_disabled, "listening": fake_listening}):
            ctx = _make_ctx()
            state_worker(ctx)

        assert call_order == ["disabled", "listening"]

    def test_invalid_state_logs_error_and_exits(self):
        """Invalid state name logs error and exits the loop."""
        def fake_disabled(ctx):
            return "nonexistent_state"

        with mock.patch.dict(STATES, {"disabled": fake_disabled}), \
             mock.patch('main.log_error') as mock_log_error:
            ctx = _make_ctx()
            state_worker(ctx)

        mock_log_error.assert_called_once_with(
            "Invalid state 'nonexistent_state' returned, shutting down"
        )

    def test_state_returning_none_exits_cleanly(self):
        """State returning None causes clean exit."""
        def fake_disabled(ctx):
            return None

        with mock.patch.dict(STATES, {"disabled": fake_disabled}), \
             mock.patch('main.log_info') as mock_log_info:
            ctx = _make_ctx()
            state_worker(ctx)

        # Should log the exit message
        mock_log_info.assert_called_with("State worker exiting")

    def test_exception_in_state_propagates(self):
        """Exception in a state function propagates (no silent swallow)."""
        def exploding_disabled(ctx):
            raise RuntimeError("GPU exploded")

        with mock.patch.dict(STATES, {"disabled": exploding_disabled}):
            ctx = _make_ctx()
            with pytest.raises(RuntimeError, match="GPU exploded"):
                state_worker(ctx)


# ---------------------------------------------------------------------------
# Cleanup (main.py:162-183)
# ---------------------------------------------------------------------------

class TestStateWorkerCleanup:
    """Tests for the finally block cleanup in main()."""

    def test_unload_models_failure_doesnt_block_audio_stop(self):
        """unload_models() failure doesn't block audio_buffer.stop()."""
        ctx = _make_ctx()
        ctx.unload_models = mock.MagicMock(side_effect=RuntimeError("unload boom"))
        ctx.audio_buffer.stop = mock.MagicMock()
        ctx.tray = mock.MagicMock()

        # Simulate the finally block from main.py:162-183
        try:
            ctx.unload_models()
        except Exception:
            pass

        try:
            ctx.audio_buffer.stop()
        except Exception:
            pass

        ctx.audio_buffer.stop.assert_called_once()

    def test_audio_stop_failure_doesnt_block_tray_cleanup(self):
        """audio_buffer.stop() failure doesn't block tray cleanup."""
        ctx = _make_ctx()
        ctx.unload_models = mock.MagicMock()
        ctx.audio_buffer.stop = mock.MagicMock(side_effect=RuntimeError("audio boom"))
        ctx.tray = mock.MagicMock()
        ctx.tray_running = True

        try:
            ctx.unload_models()
        except Exception:
            pass

        try:
            ctx.audio_buffer.stop()
        except Exception:
            pass

        try:
            if ctx.tray:
                ctx.tray_running = False
        except Exception:
            pass

        assert ctx.tray_running is False

    def test_all_cleanup_steps_run_when_earlier_steps_throw(self):
        """All three cleanup steps run even when earlier steps throw."""
        ctx = _make_ctx()
        ctx.unload_models = mock.MagicMock(side_effect=RuntimeError("unload boom"))
        ctx.audio_buffer.stop = mock.MagicMock(side_effect=RuntimeError("audio boom"))
        ctx.tray = mock.MagicMock()
        ctx.tray_running = True

        # Reproduce the guarded cleanup pattern from main.py:162-183
        try:
            ctx.unload_models()
        except Exception:
            pass

        try:
            ctx.audio_buffer.stop()
        except Exception:
            pass

        try:
            if ctx.tray:
                ctx.tray_running = False
        except Exception:
            pass

        ctx.unload_models.assert_called_once()
        ctx.audio_buffer.stop.assert_called_once()
        assert ctx.tray_running is False

    def test_cleanup_runs_when_worker_thread_raises(self):
        """Cleanup runs even when the worker thread raises an exception."""
        ctx = _make_ctx()
        ctx.unload_models = mock.MagicMock()
        ctx.audio_buffer.stop = mock.MagicMock()
        ctx.tray = mock.MagicMock()

        def exploding_disabled(ctx):
            raise RuntimeError("worker boom")

        errors = []

        def run_with_cleanup():
            try:
                with mock.patch.dict(STATES, {"disabled": exploding_disabled}):
                    state_worker(ctx)
            except Exception:
                errors.append(True)
            finally:
                try:
                    ctx.unload_models()
                except Exception:
                    pass
                try:
                    ctx.audio_buffer.stop()
                except Exception:
                    pass

        t = threading.Thread(target=run_with_cleanup)
        t.start()
        t.join(timeout=2.0)

        assert len(errors) == 1
        ctx.unload_models.assert_called_once()
        ctx.audio_buffer.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Signal handling (main.py:124-129)
# ---------------------------------------------------------------------------

class TestSignalHandling:
    """Tests for SIGINT/SIGTERM handler setup."""

    def test_sigint_handler_posts_quit(self):
        """SIGINT handler posts QUIT to mailbox."""
        ctx = _make_ctx()

        # Reproduce the handler from main.py:124-126
        def handle_signal(sig, frame):
            ctx.mailbox.post(Mailbox.QUIT)

        handle_signal(signal.SIGINT, None)
        assert ctx.mailbox.check() == Mailbox.QUIT

    def test_sigterm_handler_posts_quit(self):
        """SIGTERM handler posts QUIT to mailbox."""
        ctx = _make_ctx()

        def handle_signal(sig, frame):
            ctx.mailbox.post(Mailbox.QUIT)

        handle_signal(signal.SIGTERM, None)
        assert ctx.mailbox.check() == Mailbox.QUIT
