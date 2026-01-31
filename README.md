# Atlas Voice - Custom Wake Word Training

Train your own wake word for [OpenWakeWord](https://github.com/dscripka/openWakeWord) using synthetic speech generation.

This repo provides everything needed to train a custom wake word model, packaged for reproducibility.

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone this repo
git clone https://github.com/<your-username>/atlas-voice.git
cd atlas-voice

# Build the training environment
docker build -t atlas-voice-training .

# Run training (GPU passthrough for TensorFlow)
docker run --gpus all -v $(pwd)/output:/output atlas-voice-training
```

### Option 2: Local Installation

```bash
# Clone and run the training script
git clone https://github.com/<your-username>/atlas-voice.git
cd atlas-voice
./train.sh
```

**Requirements:**
- Python 3.10
- NVIDIA GPU with CUDA (for training)
- ~25GB disk space for training data
- `espeak-ng`, `ffmpeg` (system packages)

## What Gets Trained

The default config trains a "Hey Atlas" wake word. Edit `hey_atlas_config.yml` to change:

- `target_phrase` - The wake word/phrase to detect
- `custom_negative_phrases` - Similar phrases to NOT activate on
- `n_samples` - Number of synthetic training samples (more = better, slower)

## Training Data

Training data is downloaded from HuggingFace:
- **ACAV100M features** (17GB) - Pre-computed negative examples
- **MUSAN music** (4.6GB) - Background audio for augmentation
- **Validation features** (177MB) - False positive testing
- **Piper TTS model** (200MB) - Synthetic speech generation

## Output

After training completes, you'll have:
- `hey_atlas.tflite` - TensorFlow Lite model for inference
- `hey_atlas.onnx` - ONNX model (alternative runtime)

Copy the `.tflite` file to `~/.local/share/openwakeword/` for use with OpenWakeWord.

## Training Steps

The training script runs three phases:

1. **Generate clips** - Create synthetic speech samples with Piper TTS
2. **Augment clips** - Add noise, reverb, pitch shifts (runs on CPU)
3. **Train model** - Train neural network (uses GPU)

## License

- Training script and configs: Apache 2.0
- ACAV100M features: CC-BY-NC-SA-4.0 (non-commercial)
- MUSAN: CC BY 4.0

**Note:** The CC-BY-NC-SA-4.0 license on ACAV100M means trained models cannot be used commercially.

## Acknowledgments

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka
- [Piper TTS](https://github.com/rhasspy/piper) by Rhasspy
- [MUSAN](https://www.openslr.org/17/) corpus
