# Atlas Voice

Always-listening voice dictation for Linux. Say **"Hey Atlas"** to start dictating — your speech is transcribed and typed into the active window. Say **"break"** to end the session. Both the end session trigger word or phrase and the "wake" word are configurable or trainable.

https://github.com/user-attachments/assets/85c22beb-d879-4396-8d40-bf280f060007

## Features

- **Wake word activation** — custom-trained "Hey Atlas" model, hands-free triggering
- **GPU-accelerated transcription** — Whisper large-v3 via faster-whisper on CUDA
- **Continuous dictation** — keeps recording after each transcription without re-triggering the wake word; detects overlapping speech and seamlessly chains utterances
- **Spoken punctuation** — say "period", "comma", "new paragraph", "open paren", etc. (45+ rules, all configurable)
- **Word replacements** — auto-correct common Whisper mishearings (e.g. "cloud" → "Claude")
- **Console & GUI typing modes** — newlines as `\n` characters (for terminals and CLI tools) or as Enter keypresses (for text editors/LibreOffice); switchable by voice command
- **Terminal safety** — detects when a terminal emulator is focused and copies to clipboard instead of typing, preventing accidental command execution
- **System tray control** — pause/resume listening, enable/disable (GPU vram management), or quit
- **Systemd user service** — starts on login, restarts on failure, manageable via `systemctl --user`
- **External configuration** — all settings in a single `settings.conf` file (INI format), no source editing required

## Requirements

- Linux (tested on Ubuntu 24.04 / Linux Mint 22.x)
- NVIDIA GPU with CUDA support (4GB+ VRAM, 8GB+ recommended)
- NVIDIA driver installed (CUDA runtime libraries are bundled)
- Python 3.12
- PulseAudio
- X11 desktop environment with system tray support
- ~6GB disk space (2.5GB venv + 3GB Whisper model)

## Installation

### From .deb package (recommended)

Download the latest `.deb` from [Releases](https://github.com/briankelley/atlas-voice/releases) and install:

```bash
sudo apt install ./atlas-voice_2.0.0.deb
```

The installer will:
1. Set up a Python virtual environment with all dependencies (including CUDA runtime)
2. Download the Whisper large-v3 model (~3GB) from huggingface.co
3. Enable and start a systemd user service

The service starts automatically on login. To manage it:

```bash
systemctl --user status atlas-voice
systemctl --user restart atlas-voice
systemctl --user stop atlas-voice
journalctl --user -u atlas-voice -f   # live logs
```

### From source

```bash
git clone https://github.com/briankelley/atlas-voice.git
cd atlas-voice

# Create venv with system site-packages (required for GTK/gi bindings)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
pip install numpy sounddevice faster-whisper openwakeword
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

# Run (Whisper model auto-downloads on first run)
python main.py
```

## Usage

| Action | How |
|--------|-----|
| Start dictating | Say **"Hey Atlas"** |
| End session | Say **"break"** (types remaining text, presses Enter) |
| Pause / Resume | Tray menu → Pause / Resume |
| Unload GPU | Tray menu → Enable / Disable (GPU) |
| Quit | Tray menu → Quit Atlas |
| Switch typing mode | Say **"switch to console"** or **"switch to gui"** |

### Continuous Dictation

After your first utterance is transcribed, Atlas stays in recording mode — just keep talking. Each pause triggers a transcription and the result is typed out, then Atlas immediately listens for more. Say **"break"** when you're done to end the session and press Enter.

### Typing Modes

| Mode | Newlines sent as | Best for |
|------|-----------------|----------|
| `console` (default) | `\n` character | Terminals, CLI tools, chat apps |
| `gui` | Enter keypress | LibreOffice, text editors, form fields |

Switch modes on the fly by saying "switch to console" or "switch to gui", or set the default in `settings.conf`.

### Spoken Punctuation

All rules are configurable in `settings.conf` under `[spoken_punctuation]`. Defaults include:

| Say | Types | | Say | Types |
|-----|-------|-|-----|-------|
| "period" | `.` | | "open paren" | `(` |
| "comma" | `,` | | "close paren" | `)` |
| "question mark" | `?` | | "open bracket" | `[` |
| "exclamation point" | `!` | | "close bracket" | `]` |
| "colon" | `:` | | "open brace" | `{` |
| "semicolon" | `;` | | "close brace" | `}` |
| "dash" | `-` | | "open quote" | `"` |
| "new line" | newline | | "close quote" | `"` |
| "new paragraph" | double newline | | "apostrophe" | `'` |
| "ellipsis" | `...` | | "ampersand" | `&` |
| "underscore" | `_` | | "asterisk" | `*` |
| "at sign" | `@` | | "hashtag" | `#` |
| "forward slash" | `/` | | "backslash" | `\` |
| "equals" | `=` | | "plus" | `+` |
| "dollar sign" | `$` | | | |

## Configuration

All settings live in `settings.conf` in the installation directory. Edit and restart the service to apply:

```bash
# Edit (installed location)
nano /usr/local/lib/atlas-voice/settings.conf

# Restart to pick up changes
systemctl --user restart atlas-voice
```

### Wake Word Detection

```ini
[wake_word]
# Detection confidence (0.0–1.0). Lower = more sensitive, more false positives.
threshold = 0.35
```

### Audio Capture

```ini
[audio]
silence_threshold = 500     # Amplitude below which audio is "silence"
silence_duration = 2.0      # Seconds of silence before ending capture
max_record_duration = 60    # Hard cap per recording chunk (seconds)
buffer_seconds = 120        # Ring buffer history (seconds of audio kept in memory)
```

### Transcription

```ini
[whisper]
device = cuda               # "cuda" or "cpu"
compute_type = float16      # "float16", "int8", or "float32"
```

### Behavior

```ini
[behavior]
auto_type = true            # Type transcribed text into the active window
beep_on_wake = true         # Play a sound when wake word detected
debug_mode = false          # Verbose logging (audio health, wake scores, state transitions)
log_transcripts = false     # Log all dictated text to stdout/journald (privacy-sensitive!)
tray_enabled = true         # Show system tray icon
typing_mode = console       # "console" or "gui" (see Typing Modes above)
switch_to_console_phrase = switch to console
switch_to_gui_phrase = switch to gui
```

### Session Control

```ini
[session]
end_phrase = break          # Say this word to end session and press Enter
```

### Word Replacements

Correct common Whisper mishearings. Case-sensitive — add variants as needed:

```ini
[word_replacements]
cloud = Claude
clawed = Claude
pseudo = sudo
no help = nohup
Brake = break
```

### Spoken Punctuation

Add custom phrase-to-symbol mappings. Multi-word phrases are matched first:

```ini
[spoken_punctuation]
new paragraph = \n\n
exclamation point = !
period = .
```

## Architecture

```
                         ┌──────────────┐
                         │   main.py    │
                         │  (GTK loop)  │
                         └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
       ┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼───────┐
       │   TrayIcon  │  │   Mailbox   │  │ State Worker  │
       │ (GTK thread)│  │(thread-safe)│  │   (thread)    │
       └─────────────┘  └─────────────┘  └───────┬───────┘
                                                  │
                    ┌─────────────────────────────┤
                    │         State Machine       │
                    │                             │
              ┌─────▼─────┐                ┌─────▼─────┐
              │ disabled  │◄───────────────│  paused   │
              └─────┬─────┘                └─────▲─────┘
                    │ (load models)               │
              ┌─────▼─────┐                       │
              │ listening │───────────────────────┘
              │(wake word)│
              └─────┬─────┘
                    │ (wake detected)
              ┌─────▼─────┐
              │ recording │◄─────────────┐
              │ (capture) │              │
              └─────┬─────┘              │
                    │ (silence)          │ (continuous
              ┌─────▼──────┐             │  dictation)
              │transcribing│─────────────┘
              │ (Whisper)  │
              └────────────┘
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
    ┌─────────┐ ┌───────┐ ┌────────┐
    │xdotool  │ │xclip  │ │paplay  │
    │(typing) │ │(clip) │ │(beep)  │
    └─────────┘ └───────┘ └────────┘
```

### Module Map

| Module | Responsibility |
|--------|---------------|
| `main.py` | Entry point, signal handling, GTK main loop, state dispatch |
| `config.py` | Load and parse `settings.conf` with typed defaults |
| `context.py` | Shared state container, model load/unload, GPU memory cleanup |
| `mailbox.py` | Thread-safe request passing between GTK and worker threads |
| `audio_buffer.py` | Continuous audio capture, ring buffer, chunk queue |
| `tray.py` | System tray icon — renders icons, posts user actions to mailbox |
| `logging_utils.py` | Timestamped debug/info/error logging |
| `text_processing.py` | Spoken punctuation and word replacement pipeline |
| `text_output.py` | xdotool typing, xclip clipboard, terminal detection |
| `state_disabled.py` | Models unloaded, GPU free — waits for enable |
| `state_paused.py` | Models loaded, not listening — waits for resume |
| `state_listening.py` | Wake word detection loop with audio health checks |
| `state_recording.py` | Audio capture with silence detection and VAD mode |
| `state_transcribing.py` | Whisper inference, text output, continuous dictation |

## Training Your Own Wake Word

Want to use a different wake word? See [atlas-voice-training](https://github.com/briankelley/atlas-voice-training) for the dockerized training pipeline.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Acknowledgments

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — wake word detection engine
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper inference
- [Whisper](https://github.com/openai/whisper) — OpenAI's speech recognition model
