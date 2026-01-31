#!/bin/bash
# Train custom wake word model for OpenWakeWord
# https://github.com/dscripka/openWakeWord
#
# Usage: ./train.sh [config.yml]
# Default config: hey_atlas_config.yml

set -e
cd "$(dirname "$0")"

CONFIG="${1:-hey_atlas_config.yml}"
MODEL_NAME=$(grep "model_name:" "$CONFIG" | awk '{print $2}' | tr -d '"')

echo "=============================================="
echo "OpenWakeWord Custom Model Training"
echo "Config: $CONFIG"
echo "Model: $MODEL_NAME"
echo "=============================================="
echo ""

# Step 0: Create/activate virtual environment
echo "[Step 0/6] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "  Creating venv..."
    python3 -m venv venv
fi
source venv/bin/activate
PYTHON="$PWD/venv/bin/python3"
PIP="$PYTHON -m pip"
echo "  Python: $PYTHON"
echo "  Upgrading pip..."
$PIP install --upgrade pip wheel setuptools
echo "  Installing numpy, scipy, tqdm..."
$PIP install "numpy<2.0" scipy tqdm
echo "  [Step 0/6] DONE"
echo ""

# Step 1: Install dependencies
echo "[Step 1/6] Installing dependencies..."

# Clone openWakeWord if needed
if [ ! -d "openWakeWord" ]; then
    echo "  Cloning openWakeWord..."
    git clone https://github.com/dscripka/openWakeWord
fi

# Clone piper-sample-generator if needed
if [ ! -d "piper-sample-generator" ]; then
    echo "  Cloning piper-sample-generator (dscripka fork)..."
    git clone https://github.com/dscripka/piper-sample-generator
fi

# Download TTS model if needed
if [ ! -f "piper-sample-generator/models/en-us-libritts-high.pt" ]; then
    echo "  Downloading TTS model (~200MB)..."
    wget --show-progress -O piper-sample-generator/models/en-us-libritts-high.pt \
        'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'
fi

echo "  Installing PyTorch stack (pinned versions)..."
$PIP install torch==1.13.1 torchaudio==0.13.1

echo "  Installing TensorFlow stack (pinned versions)..."
$PIP install protobuf==3.20.3 tensorflow-cpu==2.8.1 tensorflow_probability==0.16.0 onnx==1.14.0 onnx_tf==1.10.0

echo "  Installing audio processing packages..."
$PIP install piper-phonemize piper-tts espeak-phonemizer webrtcvad mutagen torchinfo==1.8.0 torchmetrics==0.11.4 \
    speechbrain==0.5.14 audiomentations==0.30.0 torch-audiomentations==0.11.0 \
    acoustics==0.2.6 pronouncing datasets==2.14.4 deep-phonemizer==0.0.19 \
    soundfile librosa pyyaml

echo "  Installing openWakeWord..."
$PIP install -e ./openWakeWord

echo "  [Step 1/6] DONE"
echo ""

# Step 2: Download room impulse responses
echo "[Step 2/6] Downloading room impulse responses..."
if [ ! -d "mit_rirs" ] || [ -z "$(ls -A mit_rirs 2>/dev/null)" ]; then
    rm -rf mit_rirs
    $PYTHON << 'EOF'
import os
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm
import datasets

output_dir = "./mit_rirs"
os.makedirs(output_dir, exist_ok=True)

print("  Downloading MIT RIRs from HuggingFace...")
rir_dataset = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)

count = 0
for row in tqdm(rir_dataset, desc="  Saving RIRs"):
    name = row['audio']['path'].split('/')[-1]
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
    count += 1

print(f"  Saved {count} room impulse responses")
EOF
else
    echo "  MIT RIRs already downloaded."
fi
echo "  [Step 2/6] DONE"
echo ""

# Step 3: Download background audio (MUSAN music from OpenSLR)
echo "[Step 3/6] Downloading background audio..."
if [ ! -d "musan_music" ] || [ -z "$(ls -A musan_music 2>/dev/null)" ]; then
    rm -rf musan_music
    echo "  Downloading MUSAN music dataset (~1GB)..."
    wget --show-progress -O musan_music.tar.gz \
        "https://www.openslr.org/resources/17/musan.tar.gz"

    echo "  Extracting music portion..."
    tar -xzf musan_music.tar.gz musan/music --strip-components=1
    mv music musan_music
    rm -rf musan musan_music.tar.gz

    echo "  Converting to 16kHz WAV..."
    $PYTHON << 'EOF'
import os
import subprocess
from tqdm import tqdm
from pathlib import Path

input_dir = Path("./musan_music")
wav_files = list(input_dir.rglob("*.wav"))
print(f"  Found {len(wav_files)} music files")

for wav_file in tqdm(wav_files, desc="  Resampling"):
    tmp = str(wav_file) + ".tmp.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(wav_file),
        "-ar", "16000", "-ac", "1", tmp
    ], capture_output=True)
    os.replace(tmp, str(wav_file))

print(f"  Converted {len(wav_files)} files to 16kHz mono")
EOF
else
    echo "  MUSAN music already downloaded."
fi
echo "  [Step 3/6] DONE"
echo ""

# Step 4: Download pre-computed features
echo "[Step 4/6] Downloading pre-computed features..."
if [ ! -f "openwakeword_features_ACAV100M_2000_hrs_16bit.npy" ]; then
    echo "  Downloading ACAV100M features (~17GB)..."
    echo "  This is the largest download - please be patient..."
    wget --show-progress \
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
else
    echo "  ACAV100M features already downloaded."
fi

if [ ! -f "validation_set_features.npy" ]; then
    echo "  Downloading validation features..."
    wget --show-progress \
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
else
    echo "  Validation features already downloaded."
fi
echo "  [Step 4/6] DONE"
echo ""

# Step 5: Download embedding models
echo "[Step 5/6] Downloading embedding models..."
MODELS_DIR="./openWakeWord/openwakeword/resources/models"
mkdir -p "$MODELS_DIR"

if [ ! -f "$MODELS_DIR/embedding_model.onnx" ]; then
    echo "  Downloading embedding_model.onnx..."
    wget --show-progress -O "$MODELS_DIR/embedding_model.onnx" \
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
    echo "  Downloading embedding_model.tflite..."
    wget --show-progress -O "$MODELS_DIR/embedding_model.tflite" \
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite"
    echo "  Downloading melspectrogram.onnx..."
    wget --show-progress -O "$MODELS_DIR/melspectrogram.onnx" \
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
    echo "  Downloading melspectrogram.tflite..."
    wget --show-progress -O "$MODELS_DIR/melspectrogram.tflite" \
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite"
else
    echo "  Embedding models already present."
fi
echo "  [Step 5/6] DONE"
echo ""

# Step 6: Train the model
echo "[Step 6/6] Training model..."
echo ""
echo "=============================================="
echo "  Training may take 30-60+ minutes"
echo "=============================================="
echo ""

echo "  [6a] Generating synthetic speech clips..."
$PYTHON openWakeWord/openwakeword/train.py --training_config "$CONFIG" --generate_clips
echo "  Synthetic clips generated."
echo ""

echo "  [6b] Augmenting clips with noise/reverb..."
# Force CPU for augmentation (avoids CUDA/cuFFT version conflicts with old torch)
CUDA_VISIBLE_DEVICES="" $PYTHON openWakeWord/openwakeword/train.py --training_config "$CONFIG" --augment_clips
echo "  Augmentation complete."
echo ""

echo "  [6c] Training neural network (GPU accelerated)..."
$PYTHON openWakeWord/openwakeword/train.py --training_config "$CONFIG" --train_model
echo "  Training complete."
echo ""

echo "  [Step 6/6] DONE"
echo ""

# Report results
MODEL_DIR="./${MODEL_NAME}_model"
if [ -f "$MODEL_DIR/${MODEL_NAME}.tflite" ]; then
    echo "=============================================="
    echo "Training complete!"
    echo "=============================================="
    echo ""
    echo "Output files:"
    echo "  $MODEL_DIR/${MODEL_NAME}.tflite"
    echo "  $MODEL_DIR/${MODEL_NAME}.onnx"
    echo ""
    echo "To use with OpenWakeWord:"
    echo "  mkdir -p ~/.local/share/openwakeword"
    echo "  cp $MODEL_DIR/${MODEL_NAME}.tflite ~/.local/share/openwakeword/"
    echo ""
else
    echo "ERROR: Model file not found at $MODEL_DIR/${MODEL_NAME}.tflite"
    echo "Training may have failed. Check output above."
    exit 1
fi
