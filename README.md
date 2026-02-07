# OpenWakeWord Custom Wake Word Training (Dockerized)

Train your own custom wake word model for [OpenWakeWord](https://github.com/dscripka/openWakeWord) (containerized).

## The Dependency Problem

This project exists because training OpenWakeWord models in 2026 is a dependency nightmare. The training pipeline requires PyTorch 1.13.1, TensorFlow 2.8.1, and dozens of other packages pinned to 2022-era versions that have since aged out of compatibility with modern Python. The `train.sh` script in this repo documents every fix discovered through days of debugging. The Docker image freezes the working environment of openWakeWord to commit 368c037 (main on February 1, 2026).

## Objective

Builds a Docker container, downloads training data, generates synthetic speech samples, augments them with common noise, trains a neural network, and outputs a model file (~200KB) that can listen for your wake word.

## Requirements

- **NVIDIA GPU** with CUDA support
- **Docker** with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **~45GB free disk space** (20GB training data + container workspace)

All packages and dependencies are handled inside the container.

## Quick Start

```bash
git clone https://github.com/brianckelley/atlas-voice.git
cd atlas-voice
./train-wakeword.sh
```

Launch the main script and it'll walk you through the process:

1. **Builds the Docker image** (first run only, cached after that)
2. **Asks if you want to proceed** with the ~20GB training data download
3. **Asks for your wake word** with advice on what works best
4. **Shows default training settings** (configurable)
5. **Launches training** inside the container
6. **Models saved** to `docker-output/`

### Output

```
═══════════════════════════════════════════════════════
  Training Complete!
  Wake word: "Hey Atlas"

  Models:
    hey_atlas.onnx    (201K)
    hey_atlas.tflite  (207K)

  Accuracy:  81.07%  (how well it tells your wake word apart from everything else)
  Recall:    62.20%  (how often it catches your wake word; higher = less repeating yourself)
  FP/hr:     1.24    (phantom activations per hour when you're not speaking the wake word)

  Output directory: /output/
═══════════════════════════════════════════════════════
```

## Using the Model

Copy the `.tflite` file to `~/.local/share/openwakeword/` for use with OpenWakeWord.

The trained model handles wake word detection only. To build a full voice input pipeline, you also need:

| Package | Purpose |
|---------|---------|
| [OpenWakeWord](https://github.com/dscripka/openWakeWord) | Loads the `.tflite` model and listens for the wake word |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Speech-to-text transcription after the wake word triggers |
| [sounddevice](https://python-sounddevice.readthedocs.io/) | Audio capture from your microphone |

## Wake Word Selection

**Use a two-word phrase.** This was a big factor in model quality across every configuration I tested. A prefix like "Hey" or "Okay" gives the model a stronger acoustic signature.

| Wake Word | Accuracy | Recall | FP/hr | Verdict |
|-----------|----------|--------|-------|---------|
| "Hey Atlas" (50k samples) | **81.10%** | **62.48%** | 2.12 | Best overall |
| "Globe Master" (50k samples) | 81.07% | 62.20% | **1.24** | Two-word phrase, consistent |
| "Hey Atlas" (100k samples) | 77.47% | 55.08% | 0.62 | More conservative, worse recall |
| "Atlas" (50k, 3 aug rounds) | 71.64% | 43.54% | 2.57 | Single word, consistently worse |
| "Atlas" (50k, 64 neurons) | 71.94% | 44.04% | 2.48 | Extra neurons didn't help |

No combination of augmentation rounds, sample count, or neuron depth made the model more accurate when the "Hey" prefix was dropped.

## Training Settings

The defaults produce the best balance of accuracy and recall based on empirical testing. You can change this before training. Hat tip to [@dscripka](https://github.com/dscripka) for great defaults.

| Setting | Default | Purpose | Testing Notes |
|---------|---------|--------------|---------------|
| Samples | 50,000 | Number of synthetic speech clips generated | Doubling to 100k didn't improve accuracy or recall |
| Augmentation rounds | 2 | Times each clip is re-processed with noise/reverb | 3 rounds produced no measurable improvement |
| Training steps | 100,000 | Neural network training iterations | 150k steps didn't improve results |
| Layer size | 32 | Neurons per hidden layer | 64 neurons produced identical results to 32 |

## Training Data

Training data is hosted on [HuggingFace](https://huggingface.co/datasets/brianckelley/atlas-voice-training-data) and downloaded automatically in standalone mode (~20GB as a single tarball).

| File | Size | Purpose |
|------|------|---------|
| ACAV100M features | 17 GB | 2,000 hours of pre-computed negative examples |
| MUSAN music | 4.6 GB | Background audio for augmentation |
| Validation features | 177 MB | False positive testing during training |
| Piper TTS model | 200 MB | Synthetic speech generation |

MIT Room Impulse Responses (~300MB) are downloaded separately from [MIT](https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip) and converted to 16kHz automatically.

## How It Works

Training runs in three phases inside the container (start to finish with defaults and broadband is ~1h on a 4090):

1. **Generate clips** - Piper TTS creates thousands of synthetic pronunciations of your wake word with varying voices, speeds, and pitch
2. **Augment clips** - Each clip is layered with room reverb, background noise, and acoustic conditions (runs on CPU)
3. **Train model** - A neural network learns to distinguish your wake word from everything else (runs on GPU)

The output is an ONNX model and a TFLite model, both under 250KB.

## Files

| File | Purpose |
|------|---------|
| `train-wakeword.sh` | What you run - interactive host wrapper (run this on your rig) |
| `container-entrypoint.sh` | Runs inside the Docker container |
| `Dockerfile.training` | Builds the training environment |
| `validate_model.py` | Compare model accuracy against test data |

### Issues encountered and fixed (all from the upstream dependency stack):

1. `torch==1.13.1` - no wheels for Python 3.12+
2. `pyarrow` - broke `datasets` API (pinned `<15.0.0`)
3. `fsspec` - broke `datasets` glob patterns (pinned `<2024.1.0`)
4. `webrtcvad` - needs C compilation, undocumented dependency on `build-essential`
5. `python3.10-venv` - version-specific package naming
6. HuggingFace download leaves `.cache` directories that break training
7. MIT RIR files nested in `16khz/` subdirectory
8. MIT RIR files are 32kHz, training expects 16kHz
9. Docker shared memory - PyTorch DataLoader needs `--shm-size=32g`
10. HuggingFace rate limiting from repeated individual file downloads
11. Training segfaults on cleanup after model is already saved (harmless)
12. Python output buffering in Docker hides progress (`PYTHONUNBUFFERED=1`)

## License

- Training scripts and configs: Apache 2.0
- ACAV100M features: CC-BY-NC-SA-4.0 (non-commercial)
- MUSAN: CC BY 4.0

**Note:** The CC-BY-NC-SA-4.0 license on ACAV100M means trained models inherit a non-commercial restriction.

## The 'Godfather' Problem

*Just when I thought I was out, they pull me back in!*

When advances in the usability of AI rekindle a love for programming you resigned from *decades ago*, your experience as full stack programmer is suddenly so dated that the only marginal value it extends is a grasp of pipelines and processes (surprisingly none of that has changed). But being amped up to dig in and no discernable skill or experience in current stacks and syntax changes your role to that of 'prompt engineer'. I'm a touch typist, not the fastest, but I'll hold my own against anyone with enough caffeine. But I'm also 'wordy'. These mix like oil and water.

tl;dr Didn't want to spend 15 minutes pounding out deep context prompts on a keyboard that I could speak in 60 seconds. But I also didn't want to say "Hey Jarvis" to my computer.

## Acknowledgments

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka
- [Piper TTS](https://github.com/rhasspy/piper) by Rhasspy
- [MUSAN](https://www.openslr.org/17/) corpus
- [MIT Room Impulse Responses](https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip)
