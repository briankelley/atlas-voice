"""Tests for tray.py - TrayIcon GTK/AppIndicator integration.

All GTK/AppIndicator/GLib dependencies are mocked throughout.
"""

import sys
from unittest import mock

# Mock GTK dependencies before importing tray
_mock_gi = mock.MagicMock()
_mock_gtk = mock.MagicMock()
_mock_appindicator = mock.MagicMock()
_mock_glib = mock.MagicMock()

# Set up gi.require_version as a no-op and gi.repository members
_mock_gi.require_version = mock.MagicMock()
_mock_gi.repository.Gtk = _mock_gtk
_mock_gi.repository.AyatanaAppIndicator3 = _mock_appindicator
_mock_gi.repository.GLib = _mock_glib

# Patch sys.modules before importing tray
_gi_patches = {
    'gi': _mock_gi,
    'gi.repository': _mock_gi.repository,
}

with mock.patch.dict(sys.modules, _gi_patches):
    # Force re-import with mocked modules
    if 'tray' in sys.modules:
        del sys.modules['tray']
    from tray import TrayIcon

from mailbox import Mailbox


def _make_tray():
    """Create a TrayIcon with a real mailbox and mock icon dir."""
    mailbox = Mailbox()
    tray = TrayIcon(mailbox, '/tmp/icons')
    return tray, mailbox


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------

class TestTrayMenuActions:
    """Tray menu items post the correct messages to mailbox."""

    def test_on_left_click_posts_toggle_pause(self):
        """_on_left_click() posts TOGGLE_PAUSE to mailbox."""
        tray, mailbox = _make_tray()
        tray._on_left_click(None)
        assert mailbox.check() == Mailbox.TOGGLE_PAUSE

    def test_on_enable_disable_posts_toggle_enable(self):
        """_on_enable_disable() posts TOGGLE_ENABLE to mailbox."""
        tray, mailbox = _make_tray()
        tray._on_enable_disable(None)
        assert mailbox.check() == Mailbox.TOGGLE_ENABLE

    def test_on_quit_posts_quit(self):
        """_on_quit() posts QUIT to mailbox."""
        tray, mailbox = _make_tray()
        tray._on_quit(None)
        assert mailbox.check() == Mailbox.QUIT


# ---------------------------------------------------------------------------
# Icon setting
# ---------------------------------------------------------------------------

class TestTrayIconSetting:
    """set_icon_by_name() behavior with and without indicator."""

    def test_set_icon_calls_indicator_when_exists(self):
        """set_icon_by_name() calls _indicator.set_icon_full() when indicator exists."""
        tray, _ = _make_tray()
        tray._indicator = mock.MagicMock()

        tray.set_icon_by_name("AV_LISTEN")

        tray._indicator.set_icon_full.assert_called_once_with("AV_LISTEN", "Atlas Voice")

    def test_set_icon_noops_when_indicator_is_none(self):
        """set_icon_by_name() no-ops when _indicator is None (pre-setup)."""
        tray, _ = _make_tray()
        assert tray._indicator is None

        # Should not raise
        tray.set_icon_by_name("AV_LISTEN")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

class TestTraySetup:
    """setup() creates indicator with correct configuration."""

    def test_setup_creates_indicator(self):
        """setup() creates indicator, sets theme path, status, and menu."""
        tray, _ = _make_tray()

        tray.setup()

        _mock_appindicator.Indicator.new.assert_called_once_with(
            "atlas-voice",
            "AV_DISABLE",
            _mock_appindicator.IndicatorCategory.APPLICATION_STATUS
        )
        indicator = _mock_appindicator.Indicator.new.return_value
        indicator.set_icon_theme_path.assert_called_once_with('/tmp/icons')
        indicator.set_status.assert_called_once_with(
            _mock_appindicator.IndicatorStatus.ACTIVE
        )
        indicator.set_menu.assert_called_once()

    def test_menu_has_expected_items(self):
        """Menu has Pause/Resume, Enable/Disable, separator, and Quit items."""
        tray, _ = _make_tray()

        # Track Gtk.MenuItem and Gtk.SeparatorMenuItem calls
        menu_items = []
        original_menu_item = _mock_gtk.MenuItem

        def track_menu_item(**kwargs):
            item = mock.MagicMock()
            item.label = kwargs.get('label', '')
            menu_items.append(('item', item.label))
            return item

        def track_separator():
            sep = mock.MagicMock()
            menu_items.append(('separator', ''))
            return sep

        _mock_gtk.MenuItem.side_effect = track_menu_item
        _mock_gtk.SeparatorMenuItem.side_effect = track_separator
        _mock_gtk.Menu.return_value = mock.MagicMock()

        try:
            tray._build_menu()
        finally:
            _mock_gtk.MenuItem.side_effect = None
            _mock_gtk.SeparatorMenuItem.side_effect = None

        labels = [label for kind, label in menu_items if kind == 'item']
        separators = [1 for kind, _ in menu_items if kind == 'separator']

        assert "Pause / Resume" in labels
        assert "Enable / Disable (GPU)" in labels
        assert "Quit Atlas" in labels
        assert len(separators) == 1


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

class TestTrayStop:
    """stop() triggers GTK main loop quit."""

    def test_stop_calls_gtk_main_quit_via_idle_add(self):
        """stop() calls GLib.idle_add(Gtk.main_quit)."""
        tray, _ = _make_tray()

        tray.stop()

        _mock_glib.idle_add.assert_called_with(_mock_gtk.main_quit)
