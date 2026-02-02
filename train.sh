#!/bin/bash
# Train custom wake word model for OpenWakeWord
# https://github.com/dscripka/openWakeWord
#
# Usage: ./train.sh [config.yml] [--tarball]
# Default config: hey_atlas_config.yml
#
# Options:
#   --tarball    Download all training data as single archive (~20GB)
#                instead of individual files. Faster if you have good bandwidth.
#
# Training data hosted at: https://huggingface.co/datasets/brianckelley/atlas-voice-training-data

set -e
cd "$(dirname "$0")"

# =============================================================================
# Configuration
# =============================================================================
LOG_ENABLED=true                    # Set to false to disable logging
DEBUG_DIR="atlas-voice-debug"       # Directory for logs and debug output
START_TIME=$(date +%s)              # Track total runtime

# =============================================================================
# Training Parameters (adjust these for quality vs time tradeoff)
# =============================================================================
N_SAMPLES=100000                    # Synthetic training samples (default: 50000)
N_SAMPLES_VAL=10000                 # Validation samples (default: 5000)
AUGMENTATION_ROUNDS=2               # Augmentation passes per sample (default: 2)
TRAINING_STEPS=150000               # Neural network training steps (default: 100000)
# Rough time estimate: 1 hour baseline, scales ~linearly with samples/steps

# Set up debug directory and logging
mkdir -p "$DEBUG_DIR"
LOG_FILE="$DEBUG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

if [ "$LOG_ENABLED" = true ]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Logging to: $LOG_FILE"
    echo ""
fi

# HuggingFace dataset for training resources
HF_DATASET="brianckelley/atlas-voice-training-data"
HF_BASE="https://huggingface.co/datasets/${HF_DATASET}/resolve/main"

# Parse arguments
USE_TARBALL=false
CONFIG="hey_atlas_config.yml"
for arg in "$@"; do
    case $arg in
        --tarball) USE_TARBALL=true ;;
        *.yml|*.yaml) CONFIG="$arg" ;;
    esac
done

# Update config file with training parameters from script header
sed -i "s/^n_samples:.*/n_samples: $N_SAMPLES/" "$CONFIG"
sed -i "s/^n_samples_val:.*/n_samples_val: $N_SAMPLES_VAL/" "$CONFIG"
sed -i "s/^augmentation_rounds:.*/augmentation_rounds: $AUGMENTATION_ROUNDS/" "$CONFIG"
sed -i "s/^steps:.*/steps: $TRAINING_STEPS/" "$CONFIG"

MODEL_NAME=$(grep "model_name:" "$CONFIG" | awk '{print $2}' | tr -d '"')

echo "=============================================="
echo "OpenWakeWord Custom Model Training"
echo "=============================================="
echo "Started: $(date)"
echo "Config: $CONFIG"
echo "Model: $MODEL_NAME"
echo "Tarball mode: $USE_TARBALL"
echo ""
echo "Training Parameters:"
echo "  Samples: $N_SAMPLES (val: $N_SAMPLES_VAL)"
echo "  Augmentation rounds: $AUGMENTATION_ROUNDS"
echo "  Training steps: $TRAINING_STEPS"
echo "Working dir: $(pwd)"
echo ""

# System info dump
echo "[System Info]"
echo "  Hostname: $(hostname)"
echo "  OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
echo "  Kernel: $(uname -r)"
echo "  User: $(whoami)"
echo "  Shell: $SHELL"
echo ""

# GPU info
echo "[GPU Info]"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
    echo "  CUDA (nvidia-smi): $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
else
    echo "  nvidia-smi not found - no NVIDIA GPU?"
fi
if command -v nvcc &> /dev/null; then
    echo "  CUDA (nvcc): $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')"
else
    echo "  nvcc not found - CUDA toolkit not installed?"
fi
echo ""

# Disk space check
echo "[Disk Space]"
echo "  Available: $(df -h . | tail -1 | awk '{print $4}')"
echo "  Required: ~25GB for training data"
echo ""

echo "=============================================="
echo ""

# Prerequisites: Check and install system dependencies
echo "[Prerequisites] Checking system dependencies..."
MISSING_PKGS=""

if ! command -v espeak-ng &> /dev/null; then
    MISSING_PKGS="$MISSING_PKGS espeak-ng"
fi

if ! command -v ffmpeg &> /dev/null; then
    MISSING_PKGS="$MISSING_PKGS ffmpeg"
fi

# Check for build tools (needed to compile webrtcvad, etc.)
if ! command -v gcc &> /dev/null; then
    MISSING_PKGS="$MISSING_PKGS build-essential"
fi

# Determine which Python to use
# Priority: existing venv > python3.10 > python3.11 > python3 (with warning)
PYTHON_CMD=""
VENV_EXISTS=false

if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
    VENV_EXISTS=true
    VENV_PY_VERSION=$(venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "  Existing venv detected: Python $VENV_PY_VERSION"
    if [[ "$VENV_PY_VERSION" == "3.10" || "$VENV_PY_VERSION" == "3.11" ]]; then
        echo "  Venv Python version is compatible. Proceeding."
        PYTHON_CMD="python3"  # Will use venv after activation
    else
        echo "  WARNING: Existing venv uses Python $VENV_PY_VERSION (untested)"
        read -p "  Delete venv and recreate with compatible Python? [Y/n] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            rm -rf venv
            VENV_EXISTS=false
        fi
    fi
fi

if [ "$VENV_EXISTS" = false ]; then
    # No venv - find best available Python
    if command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
        echo "  Found python3.10 - will use for venv"
        # Check if python3.10-venv is installed
        if ! $PYTHON_CMD -c "import ensurepip" 2>/dev/null; then
            echo "  python3.10-venv not installed"
            MISSING_PKGS="$MISSING_PKGS python3.10-venv"
        fi
        # Check if python3.10-dev is installed (needed for compiling C extensions)
        if [ ! -f "/usr/include/python3.10/Python.h" ]; then
            echo "  python3.10-dev not installed"
            MISSING_PKGS="$MISSING_PKGS python3.10-dev"
        fi
    elif command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        echo "  Found python3.11 - will use for venv"
        # Check if python3.11-venv is installed
        if ! $PYTHON_CMD -c "import ensurepip" 2>/dev/null; then
            echo "  python3.11-venv not installed"
            MISSING_PKGS="$MISSING_PKGS python3.11-venv"
        fi
        # Check if python3.11-dev is installed (needed for compiling C extensions)
        if [ ! -f "/usr/include/python3.11/Python.h" ]; then
            echo "  python3.11-dev not installed"
            MISSING_PKGS="$MISSING_PKGS python3.11-dev"
        fi
    else
        # No python3.10 or python3.11 found
        PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        echo "  System Python: $PY_VERSION (not compatible)"
        echo ""
        echo "  =============================================="
        echo "  PYTHON VERSION ISSUE"
        echo "  =============================================="
        echo ""
        echo "  This script requires Python 3.10 or 3.11."
        echo "  Your system has Python $PY_VERSION which is NOT compatible."
        echo ""
        echo "  PyTorch 1.13.1 and TensorFlow 2.8.1 do not have wheels"
        echo "  for Python $PY_VERSION. Training WILL fail."
        echo ""

        # Check if we can offer auto-install (apt-based systems)
        if command -v apt-get &> /dev/null && command -v add-apt-repository &> /dev/null; then
            echo "  OPTION 1: Auto-install Python 3.10 (recommended)"
            echo "    This will add the deadsnakes PPA and install Python 3.10"
            echo ""
            echo "  OPTION 2: Exit and install manually"
            echo "    sudo add-apt-repository ppa:deadsnakes/ppa"
            echo "    sudo apt update"
            echo "    sudo apt install python3.10 python3.10-venv python3.10-dev"
            echo ""
            echo "  OPTION 3: Proceed with Python $PY_VERSION (will likely fail)"
            echo ""
            echo "  =============================================="
            echo ""
            read -p "  Install Python 3.10 automatically? [y/N/q] " -n 1 -r
            echo ""

            if [[ $REPLY =~ ^[Qq]$ ]]; then
                echo "  Exiting."
                exit 1
            elif [[ $REPLY =~ ^[Yy]$ ]]; then
                # User explicitly approved - auto-install python3.10
                echo ""
                echo "  Adding deadsnakes PPA..."
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                echo "  Updating package lists..."
                sudo apt-get update -qq
                echo "  Installing Python 3.10..."
                sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
                echo "  Python 3.10 installed."
                echo ""
                PYTHON_CMD="python3.10"
                # Skip to end of this block
            else
                echo ""
                read -p "  Proceed anyway with Python $PY_VERSION? [y/N] " -n 1 -r
                echo ""
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo "  Exiting. Install Python 3.10 and try again."
                    exit 1
                fi
                echo "  Continuing with Python $PY_VERSION (you were warned)..."
                PYTHON_CMD="python3"
            fi
        else
            # Non-apt system, can't auto-install
            echo "  Cannot auto-install on this system (no apt)."
            echo "  Please install Python 3.10 or 3.11 manually."
            echo ""
            echo "  =============================================="
            echo ""
            read -p "  Proceed anyway with Python $PY_VERSION? [y/N] " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "  Exiting."
                exit 1
            fi
            echo "  Continuing with Python $PY_VERSION (you were warned)..."
            PYTHON_CMD="python3"
        fi

        # Check venv/dev for whatever Python we ended up with
        if [ "$PYTHON_CMD" = "python3" ]; then
            if ! $PYTHON_CMD -c "import ensurepip" 2>/dev/null; then
                MISSING_PKGS="$MISSING_PKGS python3-venv python${PY_VERSION}-venv"
            fi
            if [ ! -f "/usr/include/python${PY_VERSION}/Python.h" ]; then
                MISSING_PKGS="$MISSING_PKGS python${PY_VERSION}-dev"
            fi
        fi
    fi
fi

if [ -n "$MISSING_PKGS" ]; then
    echo "  Missing packages:$MISSING_PKGS"
    echo "  Attempting to install..."

    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y $MISSING_PKGS
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y $MISSING_PKGS
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm $MISSING_PKGS
    else
        echo ""
        echo "ERROR: Could not auto-install packages. Please install manually:"
        echo "  $MISSING_PKGS"
        echo ""
        exit 1
    fi

    echo "  System packages installed."
else
    echo "  All system dependencies present."
fi
echo ""

# Step 0: Create/activate virtual environment
echo "[Step 0/6] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "  Creating venv with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
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

# Ask about download method if not specified via --tarball flag
if [ "$USE_TARBALL" = false ]; then
    echo "=============================================="
    echo "Download Method"
    echo "=============================================="
    echo "Training data can be downloaded as:"
    echo "  1) Individual files (~25GB total, downloads only what's needed)"
    echo "  2) Single tarball (~20GB archive, faster if you have good bandwidth)"
    echo ""
    read -p "Use tarball method? [y/N]: " tarball_choice
    if [[ "$tarball_choice" =~ ^[Yy]$ ]]; then
        USE_TARBALL=true
    fi
    echo ""
fi

# Tarball option: download everything in one archive
if [ "$USE_TARBALL" = true ]; then
    echo "[Tarball Mode] Downloading all training data as single archive..."
    if [ ! -f "atlas-voice-training-data.tar.gz" ]; then
        echo "  Downloading tarball (~20GB)..."
        wget -nv -O atlas-voice-training-data.tar.gz \
            "${HF_BASE}/archive/atlas-voice-training-data.tar.gz"
    fi

    echo "  Extracting tarball..."
    tar -xzf atlas-voice-training-data.tar.gz

    echo "  Tarball extracted. Continuing with dependency installation..."
    echo ""
fi

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
    # Patch: change debug logging to info so batch progress is visible
    sed -i 's/_LOGGER.debug/_LOGGER.info/' piper-sample-generator/generate_samples.py
fi

# Download TTS model if needed
if [ ! -f "piper-sample-generator/models/en-us-libritts-high.pt" ]; then
    echo "  Downloading TTS model (~200MB)..."
    mkdir -p piper-sample-generator/models
    wget -nv -O piper-sample-generator/models/en-us-libritts-high.pt \
        "${HF_BASE}/piper_tts_model/en-us-libritts-high.pt"
fi

echo "  Installing PyTorch stack (pinned versions)..."
echo "    torch==1.13.1 torchaudio==0.13.1"
$PIP install torch==1.13.1 torchaudio==0.13.1
echo "  Verifying PyTorch..."
$PYTHON -c "import torch; print(f'    torch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')"

echo "  Installing TensorFlow stack (pinned versions)..."
echo "    tensorflow-cpu==2.8.1 protobuf==3.20.3"
$PIP install protobuf==3.20.3 tensorflow-cpu==2.8.1 tensorflow_probability==0.16.0 onnx==1.14.0 onnx_tf==1.10.0
echo "  Verifying TensorFlow..."
$PYTHON -c "import tensorflow as tf; print(f'    tensorflow {tf.__version__}')" 2>/dev/null || echo "    TensorFlow import warning (may be OK)"

echo "  Installing audio processing packages..."
$PIP install piper-phonemize piper-tts espeak-phonemizer webrtcvad mutagen torchinfo==1.8.0 torchmetrics==0.11.4 \
    speechbrain==0.5.14 audiomentations==0.30.0 torch-audiomentations==0.11.0 \
    acoustics==0.2.6 pronouncing "datasets==2.14.4" "pyarrow<15.0.0" "fsspec<2024.1.0" deep-phonemizer==0.0.19 \
    soundfile librosa pyyaml
echo "  Audio packages installed."

echo "  Installing openWakeWord..."
$PIP install -e ./openWakeWord
echo "  Verifying openWakeWord..."
$PYTHON -c "import openwakeword; print(f'    openwakeword installed')"

echo ""
echo "  [Installed packages summary]"
$PIP list | grep -E "torch|tensorflow|openwakeword|speechbrain|piper" || true
echo ""

echo "  [Step 1/6] DONE"
echo ""

# Step 2: Download room impulse responses
echo "[Step 2/6] Downloading room impulse responses..."
if [ ! -d "mit_rirs" ] || [ -z "$(ls -A mit_rirs 2>/dev/null)" ]; then
    rm -rf mit_rirs
    mkdir -p mit_rirs
    echo "  Downloading MIT RIRs from HuggingFace..."
    # Use snapshot_download with tqdm progress (replaces deprecated huggingface-cli)
    $PYTHON << 'EOF'
from huggingface_hub import snapshot_download
import shutil
import os

repo_id = "davidscripka/MIT_environmental_impulse_responses"
temp_dir = "./mit_rirs_temp"

# Download entire dataset with progress bar
print("  Downloading (tqdm progress below)...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=temp_dir,
    local_dir_use_symlinks=False
)

# Move WAV files to mit_rirs and count them
wav_count = 0
for root, dirs, files in os.walk(temp_dir):
    for f in files:
        if f.endswith('.wav'):
            src = os.path.join(root, f)
            shutil.move(src, f"./mit_rirs/{f}")
            wav_count += 1

# Clean up temp directory
shutil.rmtree(temp_dir, ignore_errors=True)
print(f"  Saved {wav_count} room impulse responses")
EOF
    # Flatten directory structure - files may be in subdirectories like 16khz/
    # Training code expects WAV files directly in mit_rirs/
    find mit_rirs -name "*.wav" -type f -exec mv {} mit_rirs/ \; 2>/dev/null || true
    find mit_rirs -type d -empty -delete 2>/dev/null || true
    # Clean up huggingface cache directory (training code tries to load it as audio)
    rm -rf mit_rirs/.cache
else
    echo "  MIT RIRs already downloaded."
fi
# Also clean up any .cache or subdirectories that might exist from previous runs
rm -rf mit_rirs/.cache 2>/dev/null || true
find mit_rirs -type d -empty -delete 2>/dev/null || true
echo "  [Step 2/6] DONE"
echo ""

# Step 3: Download background audio (MUSAN music - pre-processed 16kHz)
echo "[Step 3/6] Downloading background audio..."
if [ ! -d "musan_music" ] || [ -z "$(ls -A musan_music 2>/dev/null)" ]; then
    rm -rf musan_music
    echo "  Downloading MUSAN music (pre-processed 16kHz, ~4.6GB)..."
    # Use snapshot_download with tqdm progress (replaces deprecated huggingface-cli)
    $PYTHON << EOF
from huggingface_hub import snapshot_download

repo_id = "${HF_DATASET}"

print("  Downloading (tqdm progress below)...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns="musan_music/**",
    local_dir=".",
    local_dir_use_symlinks=False
)
print("  MUSAN music downloaded.")
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
    wget --progress=bar:force:noscroll \
        "${HF_BASE}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
else
    echo "  ACAV100M features already downloaded."
fi

if [ ! -f "validation_set_features.npy" ]; then
    echo "  Downloading validation features (~177MB)..."
    wget -nv \
        "${HF_BASE}/validation_set_features.npy"
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
    wget -nv -O "$MODELS_DIR/embedding_model.onnx" \
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
    echo "  Downloading embedding_model.tflite..."
    wget -nv -O "$MODELS_DIR/embedding_model.tflite" \
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite"
    echo "  Downloading melspectrogram.onnx..."
    wget -nv -O "$MODELS_DIR/melspectrogram.onnx" \
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
    echo "  Downloading melspectrogram.tflite..."
    wget -nv -O "$MODELS_DIR/melspectrogram.tflite" \
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
echo "  Config file contents:"
echo "  ----------------------"
cat "$CONFIG" | sed 's/^/    /'
echo "  ----------------------"
echo ""

echo "  [6a] Generating synthetic speech clips..."
echo "       Started: $(date)"
echo "       This creates TTS samples of the wake word with variations..."
$PYTHON openWakeWord/openwakeword/train.py --training_config "$CONFIG" --generate_clips
echo "       Finished: $(date)"
echo "  Synthetic clips generated."
if [ -d "${MODEL_NAME}_model" ]; then
    echo "  Output dir contents:"
    ls -la "${MODEL_NAME}_model"/ 2>/dev/null | head -10 || true
fi
echo ""

echo "  [6b] Augmenting clips with noise/reverb..."
echo "       Started: $(date)"
echo "       CUDA_VISIBLE_DEVICES=\"\" (forcing CPU to avoid cuFFT conflicts)"
echo "       This adds background noise, reverb, pitch shifts..."
CUDA_VISIBLE_DEVICES="" $PYTHON openWakeWord/openwakeword/train.py --training_config "$CONFIG" --augment_clips
echo "       Finished: $(date)"
echo "  Augmentation complete."
echo ""

echo "  [6c] Training neural network (GPU accelerated)..."
echo "       Started: $(date)"
echo "       This trains the wake word detection model..."
$PYTHON openWakeWord/openwakeword/train.py --training_config "$CONFIG" --train_model || true
echo "       Finished: $(date)"

# Check if model was saved (openWakeWord sometimes segfaults during cleanup AFTER saving)
MODEL_FILE="${MODEL_NAME}_model/${MODEL_NAME}.onnx"
if [ -f "$MODEL_FILE" ]; then
    echo "  Training complete. Model saved: $MODEL_FILE ($(du -h "$MODEL_FILE" | cut -f1))"
else
    echo "  ERROR: Training failed - model file not found: $MODEL_FILE"
    exit 1
fi

# Convert ONNX to TFLite for broader compatibility
TFLITE_FILE="${MODEL_NAME}_model/${MODEL_NAME}.tflite"
echo "  Converting ONNX to TFLite..."
$PYTHON << EOF
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load ONNX model
onnx_model = onnx.load("${MODEL_FILE}")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("${MODEL_NAME}_model/${MODEL_NAME}_tf")

# Convert TensorFlow to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("${MODEL_NAME}_model/${MODEL_NAME}_tf")
tflite_model = converter.convert()

# Save TFLite model
with open("${TFLITE_FILE}", "wb") as f:
    f.write(tflite_model)

print(f"  TFLite model saved: ${TFLITE_FILE}")
EOF

# Clean up intermediate TF model
rm -rf "${MODEL_NAME}_model/${MODEL_NAME}_tf"

if [ -f "$TFLITE_FILE" ]; then
    echo "  TFLite conversion complete: $TFLITE_FILE ($(du -h "$TFLITE_FILE" | cut -f1))"
else
    echo "  WARNING: TFLite conversion failed (ONNX model still available)"
fi
echo ""

echo "  [Step 6/6] DONE"
echo ""

# Report results
MODEL_DIR="./${MODEL_NAME}_model"
echo ""
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
RUNTIME_MIN=$((RUNTIME / 60))
RUNTIME_SEC=$((RUNTIME % 60))

echo "=============================================="
echo "TRAINING SUMMARY"
echo "=============================================="
echo "Finished: $(date)"
echo "Total runtime: ${RUNTIME_MIN}m ${RUNTIME_SEC}s"
echo ""

if [ -f "$MODEL_DIR/${MODEL_NAME}.tflite" ]; then
    echo "STATUS: SUCCESS"
    echo ""
    echo "Output files:"
    ls -la "$MODEL_DIR/"
    echo ""
    echo "Model sizes:"
    echo "  $(du -h "$MODEL_DIR/${MODEL_NAME}.tflite" 2>/dev/null || echo 'tflite not found')"
    echo "  $(du -h "$MODEL_DIR/${MODEL_NAME}.onnx" 2>/dev/null || echo 'onnx not found')"
    echo ""
    echo "To use with OpenWakeWord:"
    echo "  mkdir -p ~/.local/share/openwakeword"
    echo "  cp $MODEL_DIR/${MODEL_NAME}.tflite ~/.local/share/openwakeword/"
    echo ""
    echo "Log file: $LOG_FILE"
    echo "=============================================="
else
    echo "STATUS: FAILED"
    echo ""
    echo "ERROR: Model file not found at $MODEL_DIR/${MODEL_NAME}.tflite"
    echo ""
    echo "Debugging info:"
    echo "  Check log file: $LOG_FILE"
    echo "  Look for errors above"
    echo "  Common issues:"
    echo "    - CUDA version mismatch"
    echo "    - Out of memory (GPU or system)"
    echo "    - Python version incompatibility"
    echo "    - Missing system packages"
    echo ""
    echo "Model directory contents (if any):"
    ls -la "$MODEL_DIR/" 2>/dev/null || echo "  (directory not found)"
    echo ""
    echo "=============================================="
    exit 1
fi
