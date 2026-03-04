# Functional Specification: Atlas Voice

## State Machine (main.py)

- **States:** `disabled` | `paused` | `listening` | `recording` | `transcribing`
- **Dispatch:** `STATES[state_name](ctx)` returns next state string; `None` exits loop
- **Initial state:** `"disabled"` (always)
- **Invalid state:** `log_error` then `state = None` (graceful shutdown)
- **Thread model:** single non-daemon worker thread serializes all transitions; GTK main loop on main thread; sounddevice callback on audio thread
- **Shutdown:** `worker.join(timeout=2.0)`; logs error if still alive; `finally` block guards each cleanup step individually with try/except
- **CUDA bootstrap:** if `_ATLAS_CUDA_READY` not in env and venv nvidia libs exist, set `LD_LIBRARY_PATH` and `os.execv` re-exec; env var prevents infinite loop
- **Env forcing:** `DISPLAY` forced to `:0` if unset
- **Signals:** SIGINT/SIGTERM post `Mailbox.QUIT`
- **Startup cleanup:** `cleanup_orphaned_temp_files()` deletes `atlas_voice_*.wav` from `XDG_RUNTIME_DIR` or `tempfile.gettempdir()`; per-file + outer try/except
- **Startup race:** `ctx._pending_icon` applied to tray after tray creation if set before tray exists

## Config (config.py)

- **Source:** `settings.conf` (INI, configparser); absence non-fatal (defaults used)
- **Case preservation:** `optionxform = str`
- **Hardcoded:** `sample_rate=16000`, `chunk_size=1280` (80ms)
- **Path resolution:** model paths, icon dir derived from `__file__` base dir
- **Beep sound:** first-found from ordered candidate list (`audio-volume-change.oga`, `bell.oga`)
- **`_build_punctuation_rules`:** [spoken_punctuation] items sorted by phrase length desc (longest match first); regex: `r',?\s*\b' + escaped + r'\b[,.]?\s*'`; escape sequences `\\n\\n` -> `\n\n`, `\\n` -> `\n`, `\\\\` -> `\\`; spacing by punctuation category (`.!?` -> suffix space; `,;:` -> suffix space; `([{` -> prefix space; `)]}` -> suffix space; `=-+` -> both; `...` -> suffix space)
- **`_build_word_replacements`:** [word_replacements] section -> direct dict mapping

## Context (context.py)

- **Inter-state fields:** `wake_time: float|None`, `captured_audio: np.ndarray|None`, `recording_mode: "wake"|"vad"|None`
- **`clear_interstate_data()`:** resets all three to None
- **`set_icon(name)`:** stores `_pending_icon`, then `GLib.idle_add(tray.set_icon_by_name, name)` if tray exists and running; no lock on `_pending_icon` (startup ordering makes this safe)
- **`load_models()`:** lazy-imports `openwakeword.model.Model`, `faster_whisper.WhisperModel`; loads wake (tflite framework) then whisper; **atomic** - whisper failure after wake success triggers `unload_models()` then re-raise
- **`unload_models()`:** del refs, `gc.collect()`, `torch.cuda.empty_cache()`; **idempotent, no-throw**; each model deletion individually guarded
- **`play_beep()`:** non-blocking `subprocess.Popen(['paplay', beep_sound])`; gated on `beep_on_wake` config and file existence
- **`handle_quit(ctx)`:** unload_models + audio_buffer.stop + tray.stop (each guarded); returns `None`

## AudioBuffer (audio_buffer.py)

- **Dual-path architecture:**
  - Ring buffer: `deque(maxlen=buffer_seconds * sample_rate/chunk_size)`, stores `(timestamp, chunk)` tuples, protected by `_lock`
  - Chunk queue: `Queue(maxsize=100)` (~8s at 80ms), inherently thread-safe
  - Audio callback writes to BOTH; `flush_chunk_queue()` does NOT clear ring buffer

- **Audio callback (`_audio_callback`):**
  - float32 `[-1,1]` -> int16 `[-32768,32767]` via `(indata[:,0] * 32768).astype(np.int16)`
  - Error-status frames: appended to sliding error window, NOT enqueued, log throttled to 1/5s
  - Clean frames: prune old errors from window, signal `_first_callback_event` on first clean cb
  - Queue overflow: `get_nowait` (drop oldest) then `put_nowait` (enqueue new)
  - `chunk.copy()` for both ring buffer and chunk queue (independent arrays)
  - Callback exceptions treated as disconnect signal (appended to error window)

- **Device resolution (`_resolve_device`):** exact match -> case-insensitive substring -> ambiguous (>1 match) raises ValueError -> not-found raises ValueError; filters to `max_input_channels > 0`

| Method | Behavior | Error handling |
|---|---|---|
| `start()` | Idempotent (noop if stream exists); resolves device fresh; resets error tracking; opens `sd.InputStream(float32, 1ch)` | Failure -> `self.stop()` -> re-raise (clean state) |
| `stop(force=False)` | Idempotent, no-throw; `force=True` uses `abort()` vs `stop()`; stop/abort + close individually guarded | Silent catch on each sub-step |
| `restart()` | `stop(force=True)` then `start()`; never raises | Returns `bool`; failure leaves clean stopped state |
| `is_healthy()` | Three modes: (1) callback never fired, (2) stale >2s, (3) error rate >80% in 5s sliding window | Returns `False` for any mode |
| `is_device_present()` | Calls `_resolve_device()`; no logging | Returns `bool` |
| `get_chunk(timeout=0.08)` | Dequeue `(ts, chunk)` from chunk_queue; stores `ts` in `last_dequeued_ts` | Returns `None` on empty |
| `flush_chunk_queue()` | Drain all pending; resets `last_dequeued_ts = None` | - |
| `get_audio_since(ts)` | Concatenate ring buffer chunks where `ts >= timestamp`; returns `np.array([], dtype=np.int16)` if none | Lock-protected |
| `has_audio_since(ts)` | Check `ts >= ring_buffer[0][0]` | Returns `False` if buffer empty |
| `detect_speech_during(start, end)` | First chunk in range with `mean(abs(chunk)) >= silence_threshold` | Returns `ts` or `None` |
| `get_speech_start_time(lookback=5.0)` | Speech onset within `time.time() - lookback` | Returns `ts` or `None` |
| `save_to_wav(audio_data, filepath)` | Write int16 mono WAV at `sample_rate` | - |

- **Concurrency:** `_lock` protects ring_buffer; `_error_lock` protects error window; `last_callback_time`/`current_amplitude` lockless (benign race); `last_dequeued_ts` single-consumer only

## Mailbox (mailbox.py)

- **Constants:** `TOGGLE_PAUSE`, `TOGGLE_ENABLE`, `QUIT`
- **Semantics:** single-slot, last-writer-wins; **QUIT is sticky** (cannot be overwritten once posted)
- **`post(req)`:** writes slot unless current is QUIT; sets `_wakeup` Event
- **`check()`:** non-blocking read-and-clear; clears both `_pending` and `_wakeup`
- **`wait(timeout)`:** blocks on `_wakeup.wait(timeout)` then calls `check()`
- **Concurrency:** `_lock` protects `_pending`; thread-safe for multiple producers, single consumer

## Tray (tray.py)

- **Backend:** AyatanaAppIndicator3, GTK 3.0
- **Icons:** `AV_DISABLE` (initial), `AV_ON`, `AV_OFF`, `AV_RECORDING`
- **Menu items:** "Pause / Resume" -> `TOGGLE_PAUSE` | "Enable / Disable (GPU)" -> `TOGGLE_ENABLE` | "Quit Atlas" -> `QUIT`
- **`setup()`:** must run on GTK thread; creates indicator + menu
- **`set_icon_by_name(name)`:** called via `GLib.idle_add` from worker thread
- **`stop()`:** `GLib.idle_add(Gtk.main_quit)`
- **Responsibility boundary:** pure UI; no state logic, model calls, or audio calls

## State: Disabled (state_disabled.py)

- **Precondition:** models unloaded, GPU free
- **Icon:** `AV_DISABLE`
- **Poll:** `mailbox.wait(timeout=0.5)`

| Input | Guard | Action | Next state |
|---|---|---|---|
| `TOGGLE_ENABLE` | device present + models load + audio starts | load_models, audio.start | `"listening"` |
| `TOGGLE_ENABLE` | device not found | log, continue | `"disabled"` |
| `TOGGLE_ENABLE` | model load fails | log, continue | `"disabled"` |
| `TOGGLE_ENABLE` | audio start fails | unload_models, continue | `"disabled"` |
| `TOGGLE_PAUSE` | - | logged, ignored | `"disabled"` |
| `QUIT` | - | handle_quit | `None` |

- **Pre-check:** device presence verified before model load (avoids wasting GPU)

## State: Paused (state_paused.py)

- **Precondition:** models loaded; audio buffer may be running
- **Icon:** `AV_OFF`
- **Poll:** `mailbox.wait(timeout=0.2)`

| Input | Action | Next state |
|---|---|---|
| `TOGGLE_PAUSE` | - | `"listening"` |
| `TOGGLE_ENABLE` | unload_models, audio.stop (silently caught) | `"disabled"` |
| `QUIT` | handle_quit | `None` |

## State: Listening (state_listening.py)

- **Precondition:** `wake_model` loaded (non-None); audio buffer may not be running (handles startup recovery)
- **Icon:** `AV_ON`
- **On entry:** `clear_interstate_data()`, `flush_chunk_queue()`, `wake_model.reset()`

- **Wake detection loop:**
  - `get_chunk(80ms)` -> `wake_model.predict(chunk)` -> check `prediction_buffer` scores against `wake_word_threshold`
  - On detection: `wake_time = last_dequeued_ts - wake_preroll`, `recording_mode = "wake"` -> `"recording"`
  - Fallback: if `last_dequeued_ts is None`, use `time.time()` with error log

- **Health checks every 2.0s:**
  - Unhealthy + restart succeeds -> continue
  - Unhealthy + restart fails + `device_name` set -> `_wait_for_device(ctx)`
  - Unhealthy + restart fails + no `device_name` -> `unload_models()` -> `"disabled"`

- **Entry with no stream:** attempt `audio.start()`; ValueError + `device_name` -> `_wait_for_device(ctx, first_boot=True)`; no `device_name` -> `"disabled"`

- **`_wait_for_device(ctx, first_boot=False)`:**
  - Force-stops audio, flushes chunk queue
  - Exponential backoff: start=2.0s, max=15.0s, jitter +/-20%
  - Mailbox checked each iteration (responsive to TOGGLE_PAUSE, TOGGLE_ENABLE, QUIT)
  - After 600s continuous polling: `unload_models()` (free GPU)
  - After 60s on `first_boot`: config validity warning
  - Recovery: device found -> `audio.start()` -> wait `_first_callback_event(2s)` -> `"_recovered"` or retry with backoff
  - On `"_recovered"` return to listening: resets health check timer, flushes queue, resets wake model

| Input | Action | Next state |
|---|---|---|
| `TOGGLE_PAUSE` | - | `"paused"` |
| `TOGGLE_ENABLE` | unload_models, audio.stop | `"disabled"` |
| `QUIT` | handle_quit | `None` |

## State: Recording (state_recording.py)

- **Precondition:** `recording_mode` is `"wake"` or `"vad"`; if wake: `wake_time` non-None; audio running
- **Icon:** `AV_RECORDING`
- **On entry:** `play_beep()`; validation: wake mode with `wake_time=None` -> clear_interstate_data -> `"listening"`

- **Wake mode:**
  1. `flush_chunk_queue()` (prevent duplication with ring buffer)
  2. `get_audio_since(ctx.wake_time)` from ring buffer
  3. Live chunks from `get_chunk()` into capture loop

- **VAD mode:**
  1. Wait for speech onset: amplitude >= `silence_threshold`
  2. `vad_timeout` seconds with no speech -> `"listening"`
  3. Mailbox checked during wait (TOGGLE_PAUSE, TOGGLE_ENABLE, QUIT)

- **Capture loop:**
  - Silence: amplitude < threshold for > `silence_duration` seconds
  - Max: `max_record_duration` seconds
  - Health checks every 2.0s
  - Output: `ctx.captured_audio = np.concatenate(audio_chunks)`
  - Zero chunks captured -> `"listening"`

| Input | Guard | Action | Next state |
|---|---|---|---|
| Silence/max | chunks > 0 | set captured_audio | `"transcribing"` |
| Silence/max | chunks == 0 | - | `"listening"` |
| Audio unhealthy | restart fails + device_name | - | `"listening"` |
| Audio unhealthy | restart fails + no device_name | unload_models | `"disabled"` |
| `TOGGLE_PAUSE` | - | discard audio | `"paused"` |
| `TOGGLE_ENABLE` | - | discard, unload, audio.stop | `"disabled"` |
| `QUIT` | - | handle_quit | `None` |

## State: Transcribing (state_transcribing.py)

- **Precondition:** `captured_audio` non-None/non-empty; `whisper_model` loaded; audio running
- **Icon:** `AV_RECORDING`

- **Pipeline:**
  1. Save `captured_audio` to temp WAV (`XDG_RUNTIME_DIR`, prefix `atlas_voice_`)
  2. `whisper_model.transcribe(temp_path, beam_size=5, vad_filter=True, language="en")`
  3. Join segment texts
  4. `strip_wake_phrase(text, config)`
  5. `_handle_mode_switch(text, ctx)` - voice commands "switch to console"/"switch to gui"
  6. `process_text(text, config)` - punctuation + word replacements
  7. `contains_break_keyword(text)` check
  8. `output_text(text + " ", ctx)` - trailing space for continuous dictation

- **Continuous dictation:**
  - `detect_speech_during(transcription_start, transcription_end)` on ring buffer
  - Overlap found: `wake_time = overlap_start`, `recording_mode = "wake"` -> `"recording"`
  - No overlap: `recording_mode = "vad"` -> `"recording"`
  - Loop: transcribe -> record -> transcribe until break keyword or VAD timeout

- **Break keyword:** remove keyword, output remaining text, `play_beep()`, `press_enter()` -> `"listening"`

- **Mailbox handling (two checkpoints):**
  - Pre-transcription: QUIT/PAUSE/ENABLE -> discard audio, transition immediately
  - Post-transcription: QUIT/PAUSE/ENABLE -> **deliver text first**, then transition

- **Audio health:** checked before AND after transcription; same recovery logic as recording
- **Temp file:** always deleted in `finally` block
- **Empty transcription:** -> `"listening"`; empty after processing -> `"listening"`
- **Transcription failure:** clear interstate data -> `"listening"`

- **`_handle_mode_switch(text, ctx)`:** regex match + strip `switch_to_console_phrase`/`switch_to_gui_phrase` from text; sets `ctx.typing_mode`; case-insensitive; trailing `[,.\s]*` consumed

## Text Processing (text_processing.py)

- **`strip_wake_phrase(text, config)`:** multi-word phrase: leading words made optional in regex (`(hey[,.\s]+)?atlas[,.\s]*`); single word: anchored at `^`; case-insensitive; count=1; strips result
- **`process_text(text, config)`:** punctuation rules applied longest-first (pre-sorted); word replacements use `\b` boundaries, case-insensitive; whitespace cleanup: collapse spaces, remove space before closing punct, remove space after opening punct
- **`contains_break_keyword(text, config)`:** fuzzy regex `\bbr[ei][ae]k\b` (catches Whisper mishearings: brick, brake, etc.); case-insensitive
- **`remove_break_keyword(text, config)`:** same pattern + trailing `[,.\s]*`

## Text Output (text_output.py)

- **`output_text(text, ctx)`:** gated on `auto_type` config; clipboard ALWAYS updated first; terminal windows (matched by WM_CLASS): clipboard only, no typing; non-terminal: type via `type_text()`
- **`type_text(text, typing_mode)`:** console mode: `xdotool type --clearmodifiers -- text`; gui mode: split on `\n`, type parts, send `Return` between; `--` prevents text parsed as flags; timeout 5s for type, 2s for key
- **`copy_to_clipboard(text)`:** `xclip -selection clipboard` with stdin; timeout 2s
- **`press_enter()`:** `xdotool key Return`; timeout 2s
- **Window class cache:** `_get_cached_window_class()` with 2.0s TTL; `TERMINAL_CLASSES` set of 14 known terminal WM_CLASS names (lowercase match)

## Logging (logging_utils.py)

- **All output:** `print(flush=True)` for systemd journal capture
- **`log_debug`:** gated on `_DEBUG`; includes `HH:MM:SS.mmm` timestamp
- **`log_info`:** unconditional
- **`log_error`:** `[ERROR]` prefix, unconditional
- **`should_log_transcripts()`:** returns `_LOG_TRANSCRIPTS`
- **Globals:** `set_debug`/`set_log_transcripts` called once at startup; read-only thereafter

## Existing Test Coverage

- **test_audio_buffer.py (15 tests):** device resolution (exact/substring/ambiguous/no-match/empty/output-only), health (high error rate, transient), queue overflow (bounded drop), start failure (clean state), restart (bool return), get_chunk timestamp storage, empty queue returns None, flush resets ts, post-overflow tuple unpacking
- **test_integration.py (4 tests):** clear_interstate_data resets, wake word int16 dtype contract, listening-to-recording full transition with mocked wake model, ring buffer preroll retrieval

## Critical Constraints

1. **Single-writer state:** all state transitions serialized on worker thread; audio callback is sole writer to ring buffer and chunk queue
2. **Audio recovery routing:** any state detecting audio failure with `device_name` configured MUST return `"listening"`; listening is sole recovery orchestrator
3. **Model atomicity:** `load_models()` rolls back on partial failure; `unload_models()` idempotent and no-throw
4. **Mailbox QUIT sticky:** once posted, cannot be overwritten; ensures shutdown cannot be lost
5. **Temp file guarantee:** transcription temp WAV always deleted in `finally`; orphaned files cleaned on startup
6. **Clipboard-first output:** clipboard always written before typing to ensure data availability even if xdotool fails
7. **Terminal safety:** terminal windows receive clipboard only (no xdotool typing) to prevent shell injection
8. **Continuous dictation invariant:** transcribing always transitions to recording (overlap->wake, no overlap->vad) unless break keyword, empty result, or mailbox interrupt
9. **Wake mode dedup:** `flush_chunk_queue()` at recording entry prevents ring buffer + queue duplication
