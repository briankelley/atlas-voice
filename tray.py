"""Atlas Voice system tray icon — GTK/AppIndicator, posts to mailbox only."""

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('AyatanaAppIndicator3', '0.1')
from gi.repository import Gtk, AyatanaAppIndicator3, GLib

from mailbox import Mailbox
from logging_utils import log_info, log_debug


class TrayIcon:
    """
    System tray icon. Responsibilities:
    1. Render whatever icon it's told (via set_icon_by_name)
    2. Post user actions to the mailbox
    3. Nothing else. No state logic. No model calls. No audio calls.
    """

    def __init__(self, mailbox, icon_dir):
        self.mailbox = mailbox
        self._icon_dir = icon_dir
        self._indicator = None

    def setup(self):
        """Create the indicator and menu. Must be called on GTK main thread."""
        self._indicator = AyatanaAppIndicator3.Indicator.new(
            "atlas-voice",
            "AV_DISABLE",
            AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS
        )
        self._indicator.set_icon_theme_path(self._icon_dir)
        self._indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
        self._indicator.set_menu(self._build_menu())
        log_info("[TRAY] Indicator created")

    def _build_menu(self):
        menu = Gtk.Menu()

        pause_item = Gtk.MenuItem(label="Pause / Resume")
        pause_item.connect("activate", self._on_left_click)
        menu.append(pause_item)

        enable_item = Gtk.MenuItem(label="Enable / Disable (GPU)")
        enable_item.connect("activate", self._on_enable_disable)
        menu.append(enable_item)

        sep = Gtk.SeparatorMenuItem()
        menu.append(sep)

        quit_item = Gtk.MenuItem(label="Quit Atlas")
        quit_item.connect("activate", self._on_quit)
        menu.append(quit_item)

        menu.show_all()
        return menu

    def _on_left_click(self, _):
        """User clicked Pause/Resume (toggle pause)."""
        log_debug("[TRAY] User clicked Pause/Resume")
        self.mailbox.post(Mailbox.TOGGLE_PAUSE)

    def _on_enable_disable(self, _):
        """User clicked Enable/Disable (toggle enable)."""
        log_debug("[TRAY] User clicked Enable/Disable")
        self.mailbox.post(Mailbox.TOGGLE_ENABLE)

    def _on_quit(self, _):
        """User clicked Quit."""
        log_debug("[TRAY] User clicked Quit")
        self.mailbox.post(Mailbox.QUIT)

    def set_icon_by_name(self, name):
        """Set the tray icon. Called via GLib.idle_add from worker thread."""
        if self._indicator:
            self._indicator.set_icon_full(name, "Atlas Voice")

    def stop(self):
        """Quit GTK main loop. Called from worker thread via GLib.idle_add."""
        GLib.idle_add(Gtk.main_quit)
