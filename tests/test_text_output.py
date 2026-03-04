"""Tests for text_output.py - terminal detection, clipboard, typing modes."""

import pytest
from unittest import mock

from text_output import (
    _is_terminal,
    TERMINAL_CLASSES,
    output_text,
    type_text,
    copy_to_clipboard,
)


class TestIsTerminal:
    """Terminal window class detection."""

    def test_known_terminals(self):
        for cls in ['gnome-terminal-server', 'kitty', 'alacritty', 'konsole', 'xterm']:
            assert _is_terminal(cls) is True

    def test_non_terminal(self):
        assert _is_terminal('firefox') is False
        assert _is_terminal('libreoffice') is False
        assert _is_terminal('code') is False

    def test_none_returns_false(self):
        assert _is_terminal(None) is False

    def test_all_terminal_classes_detected(self):
        """Every entry in TERMINAL_CLASSES is properly detected."""
        for cls in TERMINAL_CLASSES:
            assert _is_terminal(cls) is True


class TestOutputText:
    """output_text routing based on window class and config."""

    @mock.patch('text_output._get_cached_window_class', return_value='gnome-terminal-server')
    @mock.patch('text_output.copy_to_clipboard')
    @mock.patch('text_output.type_text')
    def test_terminal_copies_only(self, mock_type, mock_clip, mock_window):
        """Terminal windows get clipboard only, no typing."""
        ctx = mock.MagicMock()
        ctx.config = {'auto_type': True}
        ctx.typing_mode = 'console'

        output_text("hello", ctx)

        mock_clip.assert_called_once_with("hello")
        mock_type.assert_not_called()

    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    @mock.patch('text_output.copy_to_clipboard')
    @mock.patch('text_output.type_text')
    def test_non_terminal_types_and_copies(self, mock_type, mock_clip, mock_window):
        """Non-terminal windows get clipboard + typing."""
        ctx = mock.MagicMock()
        ctx.config = {'auto_type': True}
        ctx.typing_mode = 'console'

        output_text("hello", ctx)

        mock_clip.assert_called_once_with("hello")
        mock_type.assert_called_once_with("hello", 'console')

    @mock.patch('text_output._get_cached_window_class', return_value='firefox')
    @mock.patch('text_output.copy_to_clipboard')
    @mock.patch('text_output.type_text')
    def test_auto_type_disabled_skips_everything(self, mock_type, mock_clip, mock_window):
        """auto_type=false skips all output."""
        ctx = mock.MagicMock()
        ctx.config = {'auto_type': False}

        output_text("hello", ctx)

        mock_clip.assert_not_called()
        mock_type.assert_not_called()

    def test_empty_text_noop(self):
        """Empty text does nothing."""
        ctx = mock.MagicMock()
        output_text("", ctx)
        # Should return early, no errors


class TestTypeText:
    """xdotool typing in console and gui modes."""

    @mock.patch('text_output.subprocess.run')
    def test_console_mode_single_call(self, mock_run):
        type_text("hello world", 'console')
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert 'xdotool' in args
        assert 'type' in args
        assert 'hello world' in args

    @mock.patch('text_output.subprocess.run')
    def test_gui_mode_splits_newlines(self, mock_run):
        """GUI mode splits on newlines and sends Return keypresses."""
        type_text("line one\nline two", 'gui')
        # Should call: type "line one", key Return, type "line two"
        assert mock_run.call_count == 3

    @mock.patch('text_output.subprocess.run')
    def test_gui_mode_multiple_newlines(self, mock_run):
        type_text("a\nb\nc", 'gui')
        # type "a", key Return, type "b", key Return, type "c" = 5 calls
        assert mock_run.call_count == 5

    @mock.patch('text_output.subprocess.run')
    def test_empty_text_noop(self, mock_run):
        type_text("", 'console')
        mock_run.assert_not_called()

    @mock.patch('text_output.subprocess.run')
    def test_console_mode_preserves_newlines_in_text(self, mock_run):
        """Console mode types text as-is (including embedded newlines)."""
        type_text("hello\nworld", 'console')
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "hello\nworld" in args


class TestCopyToClipboard:
    """xclip clipboard operations."""

    @mock.patch('text_output.subprocess.run')
    def test_copies_text(self, mock_run):
        copy_to_clipboard("hello")
        mock_run.assert_called_once()
        args = mock_run.call_args
        assert 'xclip' in args[0][0]
        assert args[1]['input'] == b'hello'

    @mock.patch('text_output.subprocess.run')
    def test_empty_text_noop(self, mock_run):
        copy_to_clipboard("")
        mock_run.assert_not_called()

    @mock.patch('text_output.subprocess.run')
    def test_unicode_text(self, mock_run):
        copy_to_clipboard("hello world")
        args = mock_run.call_args
        assert args[1]['input'] == 'hello world'.encode('utf-8')
