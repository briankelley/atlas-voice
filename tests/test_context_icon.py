"""Tests for AtlasContext.set_icon() - icon dispatch and startup race handling."""

from unittest import mock

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
# Normal path (tray active)
# ---------------------------------------------------------------------------

class TestSetIconNormalPath:
    """set_icon() when tray is ready and running."""

    @mock.patch('context.GLib')
    def test_with_tray_running_calls_idle_add(self, mock_glib):
        """With tray and tray_running=True, calls GLib.idle_add()."""
        ctx = _make_ctx()
        ctx.tray = mock.MagicMock()
        ctx.tray_running = True

        ctx.set_icon("AV_LISTEN")

        mock_glib.idle_add.assert_called_once_with(
            ctx.tray.set_icon_by_name, "AV_LISTEN"
        )

    @mock.patch('context.GLib')
    def test_always_updates_pending_icon(self, mock_glib):
        """Always updates _pending_icon regardless of tray state."""
        ctx = _make_ctx()
        # No tray set
        ctx.set_icon("AV_LISTEN")
        assert ctx._pending_icon == "AV_LISTEN"

        # With tray running
        ctx.tray = mock.MagicMock()
        ctx.tray_running = True
        ctx.set_icon("AV_RECORD")
        assert ctx._pending_icon == "AV_RECORD"


# ---------------------------------------------------------------------------
# Startup race (tray not ready yet)
# ---------------------------------------------------------------------------

class TestSetIconStartupRace:
    """set_icon() during startup before tray is fully initialized."""

    @mock.patch('context.GLib')
    def test_tray_none_stores_pending_no_glib_call(self, mock_glib):
        """tray=None: stores _pending_icon, no GLib call, no exception."""
        ctx = _make_ctx()
        assert ctx.tray is None

        ctx.set_icon("AV_DISABLE")

        assert ctx._pending_icon == "AV_DISABLE"
        mock_glib.idle_add.assert_not_called()

    @mock.patch('context.GLib')
    def test_tray_set_but_not_running_stores_pending(self, mock_glib):
        """tray set but tray_running=False: stores _pending_icon, no GLib call."""
        ctx = _make_ctx()
        ctx.tray = mock.MagicMock()
        ctx.tray_running = False

        ctx.set_icon("AV_DISABLE")

        assert ctx._pending_icon == "AV_DISABLE"
        mock_glib.idle_add.assert_not_called()

    @mock.patch('context.GLib')
    def test_pending_icon_applied_after_tray_running(self, mock_glib):
        """_pending_icon applied after tray_running set True (simulating main.py:138-139)."""
        ctx = _make_ctx()
        ctx.tray = mock.MagicMock()
        ctx.tray_running = False

        # Worker sets icon before tray is running
        ctx.set_icon("AV_LISTEN")
        assert ctx._pending_icon == "AV_LISTEN"
        mock_glib.idle_add.assert_not_called()

        # main() sets tray_running and applies pending icon (main.py:135-139)
        ctx.tray_running = True
        if ctx._pending_icon:
            ctx.tray.set_icon_by_name(ctx._pending_icon)

        ctx.tray.set_icon_by_name.assert_called_once_with("AV_LISTEN")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestSetIconErrorHandling:
    """set_icon() error cases."""

    @mock.patch('context.GLib')
    def test_glib_idle_add_raises_caught_and_logged(self, mock_glib):
        """GLib.idle_add raises: exception caught, logged, no crash."""
        mock_glib.idle_add.side_effect = RuntimeError("GTK thread dead")
        ctx = _make_ctx()
        ctx.tray = mock.MagicMock()
        ctx.tray_running = True

        with mock.patch('context.log_error') as mock_log:
            # Should not raise
            ctx.set_icon("AV_LISTEN")

        mock_log.assert_called_once()
        assert "Failed to set icon" in mock_log.call_args[0][0]

    @mock.patch('context.GLib')
    def test_multiple_rapid_set_icon_last_pending_wins(self, mock_glib):
        """Multiple rapid set_icon calls: last _pending_icon wins."""
        ctx = _make_ctx()
        ctx.tray = mock.MagicMock()
        ctx.tray_running = True

        ctx.set_icon("AV_DISABLE")
        ctx.set_icon("AV_LISTEN")
        ctx.set_icon("AV_RECORD")

        assert ctx._pending_icon == "AV_RECORD"
