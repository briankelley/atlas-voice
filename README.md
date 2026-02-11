# Atlas Voice

Always-listening voice dictation for Linux with wake word activation.

Say **"Hey Atlas"** to start dictating. Your speech is transcribed and typed into the active window. Say **"break"** to end the session and press Enter.

<video src="https://atlasvoice.org/atlas-voice-demo.mp4" controls width="512"></video>

## Features

- **Wake word activation** — hands-free triggering with custom-trained "Hey Atlas" wake word
- **Continuous audio buffer** — never misses speech, even during transcription
- **GPU-accelerated transcription** — uses Whisper large-v3 on CUDA for fast, accurate results
- **Spoken punctuation** — say "period", "comma", "new line", etc.
- **System tray control** — pause/resume, enable/disable (unload GPU), or quit via tray icon
- **Systemd user service** — starts on login, restarts on failure

## Requirements

- Linux (tested on Ubuntu 24.04 / Linux Mint 22.x)
- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- NVIDIA driver installed (the .deb bundles CUDA runtime libraries, but not the driver)
- Python 3.12
- PulseAudio
- ~4GB disk space (3GB for Whisper model)

## Installation

### From .deb package (recommended)

Download the latest `.deb` from [Releases](https://github.com/briankelley/atlas-voice/releases) and install:

```bash
sudo apt install ./atlas-voice_1.0.0.deb
```

The installer will:
1. Set up a Python virtual environment with all dependencies
2. Download the Whisper large-v3 model (~3GB)
3. Install a systemd user service

After installation, log out and back in, or manually start:

```bash
systemctl --user start atlas-voice
```

### From source

```bash
git clone https://github.com/briankelley/atlas-voice.git
cd atlas-voice

# Create venv with system site-packages (for GTK/gi bindings)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
pip install numpy sounddevice faster-whisper openwakeword
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12  # For CUDA support

# Download Whisper model
# The model will be auto-downloaded on first run, or you can pre-download:
# https://huggingface.co/Systran/faster-whisper-large-v3

# Run
python atlas.py
```

## Usage

| Action | How |
|--------|-----|
| Start dictating | Say "Hey Atlas" |
| End session | Say "break" (presses Enter) |
| Pause/Resume | Tray menu → Pause / Resume |
| Disable (unload GPU) | Tray menu → Enable / Disable |
| Quit | Tray menu → Quit Atlas |

### Spoken Punctuation

| Say | Types |
|-----|-------|
| "period" | `.` |
| "comma" | `,` |
| "question mark" | `?` |
| "exclamation point" | `!` |
| "new line" | `\n` |
| "new paragraph" | `\n\n` |
| "open paren" / "close paren" | `(` / `)` |
| "colon" | `:` |
| "dash" | ` - ` |

## Configuration

All settings are in `atlas.py` near the top of the file. Edit and restart the service to apply changes.

### Wake Word Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `WAKE_WORD_THRESHOLD` | `0.35` | Detection confidence (0.0-1.0). Lower = more sensitive, more false positives |
| `WAKE_WORD_MODEL` | `models/openwakeword/hey_atlas.tflite` | Path to wake word model file |

### Audio Capture

| Variable | Default | Description |
|----------|---------|-------------|
| `SILENCE_THRESHOLD` | `500` | Amplitude below which audio is considered silence |
| `SILENCE_DURATION` | `2.0` | Seconds of silence before ending capture |
| `MAX_RECORD_DURATION` | `60` | Maximum seconds per recording chunk |
| `BUFFER_SECONDS` | `120` | Audio buffer size (seconds of audio kept in memory) |

### Transcription

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_DEVICE` | `"cuda"` | Device for inference (`"cuda"` or `"cpu"`) |
| `WHISPER_COMPUTE_TYPE` | `"float16"` | Precision (`"float16"`, `"int8"`, `"float32"`) |

### Behavior

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_TYPE` | `True` | Automatically type transcribed text into active window |
| `BEEP_ON_WAKE` | `True` | Play sound when wake word detected |
| `DEBUG_MODE` | `False` | Enable verbose logging (status every 60s, stack traces) |
| `TRAY_ENABLED` | `True` | Show system tray icon |

### Text Processing

| Variable | Description |
|----------|-------------|
| `SPOKEN_PUNCTUATION` | List of (regex, replacement) tuples for spoken punctuation |
| `WORD_REPLACEMENTS` | Dictionary of word corrections (e.g., fix common mishearings) |

## Training Your Own Wake Word

Want to use a different wake word? See [atlas-voice-training](https://github.com/briankelley/atlas-voice-training) for the dockerized training pipeline.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio Buffer   │────▶│  OpenWakeWord    │────▶│  Faster-Whisper │
│  (continuous)   │     │  (wake word)     │     │  (transcription)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │    xdotool      │
                                                 │  (type output)  │
                                                 └─────────────────┘
```

## License

MIT

## Acknowledgments

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — wake word detection
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper inference
- [Whisper](https://github.com/openai/whisper) — OpenAI's speech recognition model
