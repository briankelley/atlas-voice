"""Multi-state transition, continuous dictation, and audio recovery tests.

Tests that drive actual state functions in sequence, verifying data flows
correctly between states. Each test calls 2-4 state functions and checks
that ctx fields set by one are correctly consumed by the next.
"""

import os
import time
import threading
from unittest import mock

import numpy as np
import pytest

from audio_buffer import AudioBuffer
from context import AtlasContext
from mailbox import Mailbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(config_overrides=None):
    """Build a fully-configured ctx suitable for multi-state tests."""
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
        'wake_word_threshold': 0.35,
        'silence_duration': 0,       # instant silence for fast tests
        'max_record_duration': 60,
        'vad_timeout': 0.1,          # fast VAD timeout
        'auto_type': False,
        'beep_on_wake': False,
        'beep_sound': None,
        'debug': False,
        'log_transcripts': False,
        'tray_enabled': False,
        'typing_mode': 'console',
        'switch_to_console_phrase': '',
        'switch_to_gui_phrase': '',
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
    """Make audio buffer appear healthy and running."""
    ctx.audio_buffer.stream = mock.MagicMock()
    ctx.audio_buffer.last_callback_time = time.time()
    ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=True)


def _make_speech_chunk(amplitude=1000):
    """Create a chunk with audio above silence threshold."""
    return np.full(1280, amplitude, dtype=np.int16)


def _make_silence_chunk():
    """Create a chunk below silence threshold."""
    return np.zeros(1280, dtype=np.int16)


def _mock_get_chunk_with_ts(ctx, chunks):
    """Mock get_chunk to return (ts, chunk) pairs then None.

    Each entry is (timestamp, np.array). get_chunk stores ts in
    last_dequeued_ts and returns the array.
    """
    chunk_iter = iter(chunks)

    def side_effect(timeout=0.08):
        try:
            ts, chunk = next(chunk_iter)
            ctx.audio_buffer.last_dequeued_ts = ts
            return chunk
        except StopIteration:
            return None

    ctx.audio_buffer.get_chunk = mock.MagicMock(side_effect=side_effect)


def _mock_get_chunk(ctx, chunks):
    """Mock get_chunk to return bare chunks then None."""
    chunk_iter = iter(chunks)

    def side_effect(timeout=0.08):
        try:
            return next(chunk_iter)
        except StopIteration:
            return None

    ctx.audio_buffer.get_chunk = mock.MagicMock(side_effect=side_effect)


def _setup_wake_trigger(ctx, score=0.95):
    """Set up a mock wake model that triggers immediately."""
    wake_model = mock.MagicMock()
    wake_model.prediction_buffer = {
        'hey_atlas': [0.0, 0.0, score]
    }
    ctx.wake_model = wake_model
    return wake_model


def _make_whisper_model(text="hello world"):
    """Create a mock Whisper model returning given text."""
    model = mock.MagicMock()
    segment = mock.MagicMock()
    segment.text = text
    info = mock.MagicMock()
    model.transcribe.return_value = ([segment], info)
    return model


# ---------------------------------------------------------------------------
# Item 1: Full chain transitions
# ---------------------------------------------------------------------------

class TestFullChainTransitions:
    """Drive state functions in sequence, verify interstate data handoff."""

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_listening_to_recording_to_transcribing_to_listening(
        self, mock_clip, mock_enter, mock_output
    ):
        """Full wake cycle: listening -> recording -> transcribing -> listening."""
        from state_listening import state_listening
        from state_recording import state_recording
        from state_transcribing import state_transcribing

        ctx = _make_ctx()
        _setup_running_audio(ctx)

        # --- Phase 1: listening detects wake word ---
        _setup_wake_trigger(ctx)
        speech = _make_speech_chunk()
        ts = 1000.0
        _mock_get_chunk_with_ts(ctx, [(ts, speech)])
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        result = state_listening(ctx)
        assert result == "recording"
        assert ctx.recording_mode == "wake"
        assert ctx.wake_time is not None
        saved_wake_time = ctx.wake_time

        # --- Phase 2: recording captures audio ---
        # Seed ring buffer with audio from wake_time onward
        for i in range(5):
            chunk_ts = saved_wake_time + i * 0.08
            ctx.audio_buffer.ring_buffer.append(
                (chunk_ts, _make_speech_chunk().copy())
            )
        # Feed two silence chunks to trigger end of recording
        # (first sets silence_start, second detects silence_duration exceeded)
        _mock_get_chunk(ctx, [_make_silence_chunk(), _make_silence_chunk()])
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        result = state_recording(ctx)
        assert result == "transcribing"
        assert ctx.captured_audio is not None
        assert ctx.captured_audio.dtype == np.int16
        assert len(ctx.captured_audio) > 0

        # --- Phase 3: transcribing processes and outputs ---
        ctx.whisper_model = _make_whisper_model("hello world")
        # Mock detect_speech_during to return None (no overlap -> still returns recording)
        ctx.audio_buffer.detect_speech_during = mock.MagicMock(return_value=None)

        result = state_transcribing(ctx)
        # Continuous dictation: returns "recording" with VAD mode
        assert result == "recording"
        assert ctx.captured_audio is None  # consumed
        mock_output.assert_called_once()

    @mock.patch('state_listening.time')
    def test_disabled_to_listening_to_recording(self, mock_time):
        """Enable sequence: disabled -> listening -> recording."""
        from state_disabled import state_disabled
        from state_listening import state_listening

        mock_time.time.return_value = 1000.0
        mock_time.monotonic = time.monotonic

        ctx = _make_ctx()

        # --- Phase 1: disabled receives TOGGLE_ENABLE ---
        ctx.load_models = mock.MagicMock()
        ctx.audio_buffer.start = mock.MagicMock()
        ctx.audio_buffer.is_device_present = mock.MagicMock(return_value=True)

        # Post TOGGLE_ENABLE from a thread
        def post_enable():
            time.sleep(0.05)
            ctx.mailbox.post(Mailbox.TOGGLE_ENABLE)

        t = threading.Thread(target=post_enable)
        t.start()

        result = state_disabled(ctx)
        t.join()
        assert result == "listening"
        ctx.load_models.assert_called_once()
        ctx.audio_buffer.start.assert_called_once()

        # --- Phase 2: listening detects wake word ---
        _setup_running_audio(ctx)
        _setup_wake_trigger(ctx)
        speech = _make_speech_chunk()
        _mock_get_chunk_with_ts(ctx, [(1000.0, speech)])
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        result = state_listening(ctx)
        assert result == "recording"
        assert ctx.wake_time is not None
        assert ctx.recording_mode == "wake"

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_recording_to_transcribing_break_keyword_to_listening(
        self, mock_clip, mock_enter, mock_output
    ):
        """Break keyword ends session: recording -> transcribing -> listening."""
        from state_recording import state_recording
        from state_transcribing import state_transcribing

        ctx = _make_ctx()
        _setup_running_audio(ctx)

        # Recording: set up wake mode with prerecorded audio
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0
        for i in range(3):
            ctx.audio_buffer.ring_buffer.append(
                (999.0 + i * 0.08, _make_speech_chunk().copy())
            )
        _mock_get_chunk(ctx, [_make_silence_chunk(), _make_silence_chunk()])
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        result = state_recording(ctx)
        assert result == "transcribing"
        assert ctx.captured_audio is not None

        # Transcribing: whisper returns text with break keyword
        ctx.whisper_model = _make_whisper_model("finish this break")
        ctx.play_beep = mock.MagicMock()
        ctx.audio_buffer.detect_speech_during = mock.MagicMock(return_value=None)

        result = state_transcribing(ctx)
        assert result == "listening"
        mock_enter.assert_called_once()  # press_enter on break
        ctx.play_beep.assert_called()    # session end beep
        # output_text should have been called with remaining text
        mock_output.assert_called_once()
        output_args = mock_output.call_args[0][0]
        assert "break" not in output_args.lower()

    def test_vad_mode_no_speech_returns_listening(self):
        """VAD mode with no speech onset times out -> listening."""
        from state_recording import state_recording

        ctx = _make_ctx({'vad_timeout': 0.05})
        _setup_running_audio(ctx)
        ctx.recording_mode = "vad"
        # No chunks available -> VAD loop times out
        _mock_get_chunk(ctx, [])

        result = state_recording(ctx)
        assert result == "listening"
        assert ctx.captured_audio is None


# ---------------------------------------------------------------------------
# Item 4: Continuous dictation
# ---------------------------------------------------------------------------

class TestContinuousDictation:
    """Verify continuous dictation loop: transcribing -> recording cycles."""

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_transcribe_overlap_returns_recording_wake_mode(
        self, mock_clip, mock_enter, mock_output
    ):
        """Overlap speech detected: transcribing -> recording (wake mode)."""
        from state_transcribing import state_transcribing
        from state_recording import state_recording

        ctx = _make_ctx()
        _setup_running_audio(ctx)

        # Set up transcribing with captured audio
        ctx.captured_audio = _make_speech_chunk()
        ctx.recording_mode = "wake"
        ctx.whisper_model = _make_whisper_model("hello world")

        # Simulate overlap: speech detected during transcription
        overlap_ts = 500.0
        ctx.audio_buffer.detect_speech_during = mock.MagicMock(return_value=overlap_ts)

        result = state_transcribing(ctx)
        assert result == "recording"
        assert ctx.recording_mode == "wake"
        assert ctx.wake_time == overlap_ts
        mock_output.assert_called_once()

        # Now feed that into recording - verify it reaches back from overlap_ts
        for i in range(5):
            ctx.audio_buffer.ring_buffer.append(
                (overlap_ts + i * 0.08, _make_speech_chunk().copy())
            )
        _mock_get_chunk(ctx, [_make_silence_chunk(), _make_silence_chunk()])
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        result = state_recording(ctx)
        assert result == "transcribing"
        assert ctx.captured_audio is not None
        assert len(ctx.captured_audio) > 0

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_transcribe_no_overlap_returns_recording_vad_mode(
        self, mock_clip, mock_enter, mock_output
    ):
        """No overlap: transcribing -> recording (VAD mode)."""
        from state_transcribing import state_transcribing

        ctx = _make_ctx()
        _setup_running_audio(ctx)
        ctx.captured_audio = _make_speech_chunk()
        ctx.recording_mode = "wake"
        ctx.whisper_model = _make_whisper_model("hello world")
        ctx.audio_buffer.detect_speech_during = mock.MagicMock(return_value=None)

        result = state_transcribing(ctx)
        assert result == "recording"
        assert ctx.recording_mode == "vad"

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_vad_timeout_returns_listening(
        self, mock_clip, mock_enter, mock_output
    ):
        """After VAD mode set by transcribing, recording times out -> listening."""
        from state_transcribing import state_transcribing
        from state_recording import state_recording

        ctx = _make_ctx({'vad_timeout': 0.05})  # very short
        _setup_running_audio(ctx)
        ctx.captured_audio = _make_speech_chunk()
        ctx.recording_mode = "wake"
        ctx.whisper_model = _make_whisper_model("hello")
        ctx.audio_buffer.detect_speech_during = mock.MagicMock(return_value=None)

        result = state_transcribing(ctx)
        assert result == "recording"
        assert ctx.recording_mode == "vad"

        # Recording in VAD mode: no speech arrives, times out
        _mock_get_chunk(ctx, [])  # empty - no chunks available
        result = state_recording(ctx)
        assert result == "listening"
        assert ctx.captured_audio is None

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_continuous_dictation_three_utterances(
        self, mock_clip, mock_enter, mock_output
    ):
        """Three-cycle dictation: 3x (recording -> transcribing) then VAD timeout."""
        from state_recording import state_recording
        from state_transcribing import state_transcribing

        ctx = _make_ctx({'vad_timeout': 0.05})
        _setup_running_audio(ctx)

        utterances = ["first sentence", "second sentence", "third sentence"]
        overlap_timestamps = [500.0, 600.0]  # first two have overlap

        for i, text in enumerate(utterances):
            # Recording phase
            if i == 0:
                # First recording: wake mode
                ctx.recording_mode = "wake"
                ctx.wake_time = 400.0
                for j in range(3):
                    ctx.audio_buffer.ring_buffer.append(
                        (400.0 + j * 0.08, _make_speech_chunk().copy())
                    )
            # For subsequent recordings, recording_mode/wake_time are set by
            # previous transcribing call

            _mock_get_chunk(ctx, [_make_silence_chunk(), _make_silence_chunk()])
            ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

            if ctx.recording_mode == "wake":
                # Seed ring buffer for wake mode reach-back
                if not any(ts >= ctx.wake_time for ts, _ in ctx.audio_buffer.ring_buffer):
                    for j in range(3):
                        ctx.audio_buffer.ring_buffer.append(
                            (ctx.wake_time + j * 0.08, _make_speech_chunk().copy())
                        )

            result = state_recording(ctx)
            assert result == "transcribing", f"Utterance {i}: expected transcribing"
            assert ctx.captured_audio is not None

            # Transcribing phase
            ctx.whisper_model = _make_whisper_model(text)
            if i < len(overlap_timestamps):
                # Overlap detected -> continue in wake mode
                ctx.audio_buffer.detect_speech_during = mock.MagicMock(
                    return_value=overlap_timestamps[i]
                )
            else:
                # No overlap -> VAD mode
                ctx.audio_buffer.detect_speech_during = mock.MagicMock(
                    return_value=None
                )

            result = state_transcribing(ctx)
            assert result == "recording"

        # Final recording: VAD timeout
        _mock_get_chunk(ctx, [])
        result = state_recording(ctx)
        assert result == "listening"

        # Verify output_text called 3 times
        assert mock_output.call_count == 3

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_break_keyword_mid_dictation(
        self, mock_clip, mock_enter, mock_output
    ):
        """Break keyword during third utterance ends session."""
        from state_recording import state_recording
        from state_transcribing import state_transcribing

        ctx = _make_ctx({'vad_timeout': 0.05})
        _setup_running_audio(ctx)

        utterances = ["first sentence", "second sentence", "done with this break"]
        overlap_timestamps = [500.0, 600.0]

        final_result = None
        for i, text in enumerate(utterances):
            # Recording
            if i == 0:
                ctx.recording_mode = "wake"
                ctx.wake_time = 400.0
                for j in range(3):
                    ctx.audio_buffer.ring_buffer.append(
                        (400.0 + j * 0.08, _make_speech_chunk().copy())
                    )

            if ctx.recording_mode == "wake":
                if not any(ts >= ctx.wake_time for ts, _ in ctx.audio_buffer.ring_buffer):
                    for j in range(3):
                        ctx.audio_buffer.ring_buffer.append(
                            (ctx.wake_time + j * 0.08, _make_speech_chunk().copy())
                        )

            _mock_get_chunk(ctx, [_make_silence_chunk(), _make_silence_chunk()])
            ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

            result = state_recording(ctx)
            assert result == "transcribing"

            # Transcribing
            ctx.whisper_model = _make_whisper_model(text)
            ctx.play_beep = mock.MagicMock()
            if i < len(overlap_timestamps):
                ctx.audio_buffer.detect_speech_during = mock.MagicMock(
                    return_value=overlap_timestamps[i]
                )
            else:
                ctx.audio_buffer.detect_speech_during = mock.MagicMock(
                    return_value=None
                )

            result = state_transcribing(ctx)
            final_result = result

            if result == "listening":
                break  # break keyword hit

        assert final_result == "listening"
        assert mock_output.call_count == 3
        mock_enter.assert_called_once()  # press_enter only on break


# ---------------------------------------------------------------------------
# Item 5: Audio recovery across states
# ---------------------------------------------------------------------------

class TestAudioRecoveryAcrossStates:
    """Test audio failure in one state, recovery in another."""

    @mock.patch('state_recording.time')
    @mock.patch('state_listening.time')
    @mock.patch('state_listening.random')
    def test_unhealthy_during_recording_recovers_in_listening(
        self, mock_random, mock_listen_time, mock_record_time
    ):
        """Recording -> unhealthy -> listening -> _wait_for_device -> recovered."""
        from state_recording import state_recording
        from state_listening import state_listening

        # Mock recording time: make health check fire immediately
        # recording_start=100, last_health_check=100, then loop: now=103 (>2s)
        record_times = iter([100.0, 100.0, 103.0, 103.0, 103.0])
        mock_record_time.time.side_effect = lambda: next(record_times, 103.0)

        # Mock listening time for _wait_for_device
        listen_times = iter([1000.0] * 20)
        mock_listen_time.time.side_effect = lambda: next(listen_times, 1000.0)
        mock_listen_time.monotonic = time.monotonic
        mock_random.uniform.return_value = 1.0

        ctx = _make_ctx({'audio_device': 'Test USB Mic'})
        ctx.audio_buffer.device_name = 'Test USB Mic'

        # Recording: audio goes unhealthy, restart fails
        ctx.audio_buffer.stream = mock.MagicMock()
        ctx.audio_buffer.last_callback_time = time.time()
        ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=False)
        ctx.audio_buffer.restart = mock.MagicMock(return_value=False)
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0
        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk()))
        # Feed speech chunks so the loop runs to the health check
        _mock_get_chunk(ctx, [_make_speech_chunk(), _make_speech_chunk()])
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        result = state_recording(ctx)
        assert result == "listening"

        # Now in listening: stream is None, device_name set -> _wait_for_device
        ctx.audio_buffer.stream = None
        ctx.audio_buffer.start = mock.MagicMock()  # succeeds
        ctx.audio_buffer.is_device_present = mock.MagicMock(return_value=True)
        ctx.audio_buffer._first_callback_event = mock.MagicMock()
        ctx.audio_buffer._first_callback_event.wait.return_value = True

        # After recovery, wake model triggers immediately
        _setup_wake_trigger(ctx)
        speech = _make_speech_chunk()
        _mock_get_chunk_with_ts(ctx, [(1000.0, speech)])
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=True)

        # Mock sleep for _wait_for_device to be near-instant
        original_sleep = time.sleep
        mock_listen_time.sleep = lambda s: original_sleep(0.01)

        result = state_listening(ctx)
        assert result == "recording"
        ctx.audio_buffer.start.assert_called()

    @mock.patch('state_recording.time')
    def test_unhealthy_during_recording_no_device_name_goes_disabled(
        self, mock_record_time
    ):
        """Unhealthy with no device_name -> disabled directly."""
        from state_recording import state_recording

        # Make health check fire immediately
        record_times = iter([100.0, 100.0, 103.0, 103.0, 103.0])
        mock_record_time.time.side_effect = lambda: next(record_times, 103.0)

        ctx = _make_ctx()  # no audio_device configured
        ctx.audio_buffer.stream = mock.MagicMock()
        ctx.audio_buffer.last_callback_time = time.time()
        ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=False)
        ctx.audio_buffer.restart = mock.MagicMock(return_value=False)
        ctx.unload_models = mock.MagicMock()
        ctx.recording_mode = "wake"
        ctx.wake_time = 999.0
        ctx.audio_buffer.ring_buffer.append((999.0, _make_speech_chunk()))
        # Feed speech chunks so the loop reaches the health check
        _mock_get_chunk(ctx, [_make_speech_chunk(), _make_speech_chunk()])
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        result = state_recording(ctx)
        assert result == "disabled"
        ctx.unload_models.assert_called_once()

    @mock.patch('state_transcribing.output_text')
    @mock.patch('state_transcribing.press_enter')
    @mock.patch('state_transcribing.copy_to_clipboard')
    def test_unhealthy_during_transcribing_recovers(
        self, mock_clip, mock_enter, mock_output
    ):
        """Unhealthy pre-transcription -> listening -> recovery."""
        from state_transcribing import state_transcribing

        ctx = _make_ctx({'audio_device': 'Test USB Mic'})
        ctx.audio_buffer.device_name = 'Test USB Mic'
        ctx.captured_audio = _make_speech_chunk()
        ctx.recording_mode = "wake"
        ctx.whisper_model = _make_whisper_model("should not see this")

        # Audio unhealthy, restart fails
        ctx.audio_buffer.stream = mock.MagicMock()
        ctx.audio_buffer.last_callback_time = time.time()
        ctx.audio_buffer.is_healthy = mock.MagicMock(return_value=False)
        ctx.audio_buffer.restart = mock.MagicMock(return_value=False)

        result = state_transcribing(ctx)
        assert result == "listening"
        # Transcription should NOT have been called
        ctx.whisper_model.transcribe.assert_not_called()
        mock_output.assert_not_called()
        # Interstate data cleared
        assert ctx.captured_audio is None

    @mock.patch('state_listening.time')
    @mock.patch('state_listening.random')
    def test_recovery_quit_during_device_poll(self, mock_random, mock_time):
        """QUIT posted during _wait_for_device -> clean shutdown."""
        from state_listening import state_listening

        mock_random.uniform.return_value = 1.0

        # Time: make sleep calls effectively instant
        time_values = iter([1000.0] * 20)
        mock_time.time.side_effect = lambda: next(time_values, 1000.0)
        mock_time.monotonic = time.monotonic

        ctx = _make_ctx({'audio_device': 'Test USB Mic'})
        ctx.audio_buffer.device_name = 'Test USB Mic'
        ctx.audio_buffer.stream = None  # no stream -> start() will be tried
        ctx.audio_buffer.start = mock.MagicMock(side_effect=ValueError("not found"))
        ctx.audio_buffer.is_device_present = mock.MagicMock(return_value=False)
        ctx.audio_buffer.stop = mock.MagicMock()
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()
        ctx.unload_models = mock.MagicMock()

        # Post QUIT after a brief delay
        def post_quit():
            time.sleep(0.1)
            ctx.mailbox.post(Mailbox.QUIT)

        t = threading.Thread(target=post_quit)
        t.start()

        # Mock sleep to actually sleep briefly so the thread can post
        original_sleep = time.sleep
        mock_time.sleep = lambda s: original_sleep(0.05)

        result = state_listening(ctx)
        t.join()
        assert result is None  # shutdown

    @mock.patch('state_listening.time')
    @mock.patch('state_listening.random')
    def test_recovery_toggle_pause_during_device_poll(self, mock_random, mock_time):
        """TOGGLE_PAUSE during _wait_for_device -> paused."""
        from state_listening import state_listening

        mock_random.uniform.return_value = 1.0
        time_values = iter([1000.0] * 20)
        mock_time.time.side_effect = lambda: next(time_values, 1000.0)
        mock_time.monotonic = time.monotonic

        ctx = _make_ctx({'audio_device': 'Test USB Mic'})
        ctx.audio_buffer.device_name = 'Test USB Mic'
        ctx.audio_buffer.stream = None
        ctx.audio_buffer.start = mock.MagicMock(side_effect=ValueError("not found"))
        ctx.audio_buffer.is_device_present = mock.MagicMock(return_value=False)
        ctx.audio_buffer.stop = mock.MagicMock()
        ctx.audio_buffer.flush_chunk_queue = mock.MagicMock()

        def post_pause():
            time.sleep(0.1)
            ctx.mailbox.post(Mailbox.TOGGLE_PAUSE)

        t = threading.Thread(target=post_pause)
        t.start()

        original_sleep = time.sleep
        mock_time.sleep = lambda s: original_sleep(0.05)

        result = state_listening(ctx)
        t.join()
        assert result == "paused"
