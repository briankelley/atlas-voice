"""Atlas Voice text output — xdotool/xclip with security checks."""

import subprocess
import time
from logging_utils import log_debug, log_info, log_error


# Known terminal emulator window classes
TERMINAL_CLASSES = {
    'gnome-terminal', 'gnome-terminal-server',
    'xterm', 'konsole', 'terminator', 'alacritty',
    'kitty', 'tilix', 'sakura', 'guake', 'yakuake',
    'st', 'urxvt', 'rxvt',
}

# Window class cache (avoids repeated xdotool subprocess calls)
_cached_window_class = None
_window_class_cache_time = 0
_WINDOW_CLASS_CACHE_TTL = 2.0


def _get_active_window_class():
    """Get the WM_CLASS of the currently focused window. Returns lowercase string or None."""
    try:
        result = subprocess.run(
            ['xdotool', 'getactivewindow', 'getwindowclassname'],
            capture_output=True,
            timeout=1.0,
            shell=False,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().lower()
    except subprocess.TimeoutExpired:
        log_error("[OUTPUT] xdotool window class detection timeout")
    except Exception as e:
        log_error(f"[OUTPUT] xdotool window class detection failed: {e}")
    return None


def _is_terminal(window_class):
    """Check if window class is a known terminal emulator."""
    if window_class is None:
        return False
    return window_class in TERMINAL_CLASSES


def _get_cached_window_class():
    """Get window class with TTL caching to avoid repeated subprocess calls."""
    global _cached_window_class, _window_class_cache_time
    now = time.time()
    if now - _window_class_cache_time < _WINDOW_CLASS_CACHE_TTL:
        return _cached_window_class
    _cached_window_class = _get_active_window_class()
    _window_class_cache_time = now
    return _cached_window_class


def output_text(text, ctx):
    """Output transcribed text via typing or clipboard, with security checks."""
    if not text:
        return

    if not ctx.config.get('auto_type', True):
        return

    window_class = _get_cached_window_class()
    log_debug(f"[OUTPUT] Target window: {window_class}")

    if _is_terminal(window_class):
        log_info("[OUTPUT] Terminal detected, copying to clipboard (not typing)")
        copy_to_clipboard(text)
        return

    typing_mode = ctx.typing_mode
    type_text(text, typing_mode)


def type_text(text, typing_mode='console'):
    """Type text via xdotool with security checks."""
    if not text:
        return

    try:
        if typing_mode == 'gui':
            # GUI mode: split on newlines and send actual Enter keypresses
            parts = text.split('\n')
            for i, part in enumerate(parts):
                if part:
                    subprocess.run(
                        ['xdotool', 'type', '--clearmodifiers', '--', part],
                        timeout=5.0,
                        shell=False,
                        check=True
                    )
                if i < len(parts) - 1:
                    subprocess.run(
                        ['xdotool', 'key', 'Return'],
                        timeout=2.0,
                        shell=False,
                        check=True
                    )
        else:
            # Console mode: type text as-is
            subprocess.run(
                ['xdotool', 'type', '--clearmodifiers', '--', text],
                timeout=5.0,
                shell=False,
                check=True
            )
        log_info(f"[OUTPUT] Typed {len(text)} chars")
    except subprocess.TimeoutExpired:
        log_error("[OUTPUT] xdotool type timeout")
    except subprocess.CalledProcessError as e:
        log_error(f"[OUTPUT] xdotool type failed: {e}")


def copy_to_clipboard(text):
    """Copy text to clipboard via xclip."""
    if not text:
        return
    try:
        subprocess.run(
            ['xclip', '-selection', 'clipboard'],
            input=text.encode('utf-8'),
            timeout=2.0,
            shell=False,
            check=True
        )
        log_info(f"[OUTPUT] Copied {len(text)} chars to clipboard")
    except subprocess.TimeoutExpired:
        log_error("[OUTPUT] xclip timeout")
    except subprocess.CalledProcessError as e:
        log_error(f"[OUTPUT] xclip failed: {e}")


def press_enter():
    """Press Enter key using xdotool."""
    try:
        subprocess.run(
            ['xdotool', 'key', 'Return'],
            timeout=2.0,
            shell=False,
            check=True
        )
    except Exception as e:
        log_error(f"[OUTPUT] Enter key failed: {e}")
