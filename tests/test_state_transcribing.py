"""Tests for state_transcribing.py - Whisper inference, text pipeline,
continuous dictation, and break keyword handling."""

import os
import time
import pytest
from unittest import mock

import numpy as np

from audio_buffer import AudioBuffer
from context import AtlasContext
from mailbox import Mailbox
from state_transcribing import state_transcribing, _handle_mode_switch


def _make_ctx(config_overrides=None):
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
        'beep_sound': None,
        'debug': False,
        'log_transcripts': False,
        'tray_enabled': False,
        'typing_mode': 'console',
        'switch_to_console_phrase': 'switch to console',
        'switch_to_gui_phrase': 'switch to gui',
        'spoken_punctuation': [],
        'word_replacements': {},
        'wake_phrase': 'hey atlas',
        'end_phrase': 'break',
        'end_phrase_pattern': r'\b(?:break|brake|brick)\b',
        'icon_dir': '/tmp',
    }
    if config_overrides:
        config.update(config_overrides)
    buf = AudioBuffer(config)
    ctx = AtlasContext(config, buf)
    ctx.mailbox = Mailbox()
    return ctx


def _setup_running_audio(ctx):
    ctx.audio_buffer.stream = mock.MagicMock()
    ctx.audio_buffer.last_callback_time = time.time()
    ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=True)


def _make_whisper_model(text="hello world"):
    """Create a mock Whisper model that returns the given text."""
    model = mock.MagicMock()
    segment = mock.MagicMock()
    segment.text = text
    info = mock.MagicMock()
    model.transcribe.return_value = ([segment], info)
    return model


class TestTranscribingNormalFlow:
    """Normal transcription: audio -> Whisper -> text output -> continuous dictation."""

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_normal_transcription_returns_recording(self, mock_clip, mock_output):
        """Successful transcription loops back to recording (continuous dictation)."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello world")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)

        # No overlap detected
        ctx.audio_buffer.detect_speech_during = mock.MagicMock(return_value=None)

        result = state_transcribing(ctx)

        assert result == "recording"
        assert ctx.recording_mode == "vad"  # no overlap -> VAD mode

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_overlap_detected_sets_wake_mode(self, mock_clip, mock_output):
        """Overlap detected during transcription sets wake mode for next recording."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello world")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)

        overlap_ts = 5000.0
        ctx.audio_buffer.detect_speech_during = mock.MagicMock(return_value=overlap_ts)

        result = state_transcribing(ctx)

        assert result == "recording"
        assert ctx.recording_mode == "wake"
        assert ctx.wake_time == overlap_ts

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_text_output_called(self, mock_clip, mock_output):
        """Transcribed text is output with trailing space."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello world")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)
        ctx.audio_buffer.detect_speech_during = mock.MagicMock(return_value=None)

        state_transcribing(ctx)

        mock_output.assert_called_once()
        call_text = mock_output.call_args[0][0]
        assert "hello world" in call_text
        assert call_text.endswith(" ")  # trailing space for continuous dictation


class TestTranscribingEmptyResult:
    """Empty transcription returns to listening."""

    def test_empty_transcription_returns_listening(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("   ")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)

        result = state_transcribing(ctx)
        assert result == "listening"

    def test_no_captured_audio_returns_listening(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.captured_audio = None

        result = state_transcribing(ctx)
        assert result == "listening"

    def test_empty_captured_audio_returns_listening(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.captured_audio = np.array([], dtype=np.int16)

        result = state_transcribing(ctx)
        assert result == "listening"


class TestTranscribingBreakKeyword:
    """Break keyword ends dictation session."""

    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_break_keyword_returns_listening(self, mock_clip, mock_output, mock_enter):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello break")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)

        result = state_transcribing(ctx)

        assert result == "listening"
        mock_enter.assert_called_once()

    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_break_keyword_outputs_remaining_text(self, mock_clip, mock_output, mock_enter):
        """Text before the break keyword is still output."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello world break")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)

        state_transcribing(ctx)

        mock_output.assert_called_once()
        output_text = mock_output.call_args[0][0]
        assert "hello" in output_text
        assert "break" not in output_text.lower()

    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_break_variant_brake(self, mock_clip, mock_output, mock_enter):
        """Whisper mishearing 'brake' triggers break."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("done brake")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)

        result = state_transcribing(ctx)
        assert result == "listening"
        mock_enter.assert_called_once()

    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_break_only_no_remaining_text(self, mock_clip, mock_output, mock_enter):
        """If transcription is just 'break', no text is output."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("break")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)

        state_transcribing(ctx)

        # output_text should NOT be called (no remaining text)
        mock_output.assert_not_called()
        mock_enter.assert_called_once()


class TestTranscribingWakePhrase:
    """Wake phrase is stripped from transcription."""

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_wake_phrase_stripped(self, mock_clip, mock_output):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("Hey Atlas hello world")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)
        ctx.audio_buffer.detect_speech_during = mock.MagicMock(return_value=None)

        state_transcribing(ctx)

        output_text = mock_output.call_args[0][0]
        assert "hey atlas" not in output_text.lower()
        assert "hello world" in output_text

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_wake_phrase_only_returns_listening(self, mock_clip, mock_output):
        """If transcription is just the wake phrase, treat as empty."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("Hey Atlas")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)

        result = state_transcribing(ctx)
        assert result == "listening"


class TestTranscribingMailboxInterrupts:
    """Mailbox interrupts before and after transcription."""

    def test_pre_transcription_quit(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)
        ctx.mailbox.post(Mailbox.QUIT)

        result = state_transcribing(ctx)
        assert result is None

    def test_pre_transcription_pause_discards_audio(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)
        ctx.mailbox.post(Mailbox.TOGGLE_PAUSE)

        result = state_transcribing(ctx)
        assert result == "paused"

    @mock.patch('context.gc')
    def test_pre_transcription_enable_unloads(self, mock_gc):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)
        ctx.unload_models = mock.MagicMock()
        ctx.audio_buffer.stop = mock.MagicMock()
        ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)

        result = state_transcribing(ctx)

        assert result == "disabled"
        ctx.unload_models.assert_called()

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_post_transcription_pause_delivers_text_first(self, mock_clip, mock_output):
        """Post-transcription TOGGLE_PAUSE delivers text before transitioning."""
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)

        # Build a whisper mock that posts PAUSE during transcription
        segment = mock.MagicMock()
        segment.text = "important text"
        info = mock.MagicMock()

        def transcribe_and_post(*args, **kwargs):
            ctx.mailbox.post(Mailbox.TOGGLE_PAUSE)
            return ([segment], info)

        ctx.whisper_model = mock.MagicMock()
        ctx.whisper_model.transcribe.side_effect = transcribe_and_post

        result = state_transcribing(ctx)

        assert result == "paused"
        # Text should have been delivered before pausing
        mock_output.assert_called_once()
        assert "important text" in mock_output.call_args[0][0]


class TestTranscribingTempFile:
    """Temp file is always cleaned up."""

    def test_temp_file_deleted_on_success(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)
        ctx.audio_buffer.detect_speech_during = mock.MagicMock(return_value=None)

        # Track temp files created
        created_files = []
        original_named = __import__('tempfile').NamedTemporaryFile

        def tracking_temp(**kwargs):
            f = original_named(**kwargs)
            created_files.append(f.name)
            return f

        with mock.patch('state_transcribing.tempfile.NamedTemporaryFile', side_effect=tracking_temp):
            state_transcribing(ctx)

        # All temp files should be cleaned up
        for path in created_files:
            assert not os.path.exists(path)

    def test_temp_file_deleted_on_transcription_failure(self):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = mock.MagicMock()
        ctx.whisper_model.transcribe.side_effect = RuntimeError("Whisper crashed")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)

        created_files = []
        original_named = __import__('tempfile').NamedTemporaryFile

        def tracking_temp(**kwargs):
            f = original_named(**kwargs)
            created_files.append(f.name)
            return f

        with mock.patch('state_transcribing.tempfile.NamedTemporaryFile', side_effect=tracking_temp):
            result = state_transcribing(ctx)

        assert result == "listening"
        for path in created_files:
            assert not os.path.exists(path)


class TestTranscribingAudioHealth:
    """Audio health checks before and after transcription."""

    @mock.patch('context.gc')
    def test_unhealthy_before_transcription_with_device_returns_listening(self, mock_gc):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)
        ctx.audio_buffer.device_name = "Test Device"

        ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=False)
        ctx.audio_buffer.restart = mock.MagicMock(return_value=False)

        result = state_transcribing(ctx)
        assert result == "listening"

    @mock.patch('context.gc')
    def test_unhealthy_before_transcription_no_device_returns_disabled(self, mock_gc):
        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.whisper_model = _make_whisper_model("hello")
        ctx.captured_audio = np.zeros(16000, dtype=np.int16)
        ctx.audio_buffer.device_name = ''
        ctx.unload_models = mock.MagicMock()

        ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=False)
        ctx.audio_buffer.restart = mock.MagicMock(return_value=False)

        result = state_transcribing(ctx)
        assert result == "disabled"


class TestHandleModeSwitch:
    """Voice command mode switching."""

    def test_switch_to_gui(self):
        ctx = _make_ctx()
        text = _handle_mode_switch("hello switch to gui world", ctx)
        assert ctx.typing_mode == 'gui'
        assert "switch to gui" not in text.lower()
        assert "hello" in text

    def test_switch_to_console(self):
        ctx = _make_ctx()
        ctx.typing_mode = 'gui'  # start in gui
        text = _handle_mode_switch("switch to console please", ctx)
        assert ctx.typing_mode == 'console'
        assert "switch to console" not in text.lower()

    def test_no_switch_command(self):
        ctx = _make_ctx()
        original_mode = ctx.typing_mode
        text = _handle_mode_switch("just regular text", ctx)
        assert ctx.typing_mode == original_mode
        assert text == "just regular text"

    def test_switch_case_insensitive(self):
        ctx = _make_ctx()
        _handle_mode_switch("SWITCH TO GUI", ctx)
        assert ctx.typing_mode == 'gui'

    def test_switch_strips_trailing_punctuation(self):
        ctx = _make_ctx()
        text = _handle_mode_switch("switch to gui, hello", ctx)
        assert ctx.typing_mode == 'gui'
        assert "hello" in text
