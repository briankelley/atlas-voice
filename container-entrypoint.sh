#!/bin/bash
# Training script that runs INSIDE the Docker container
# Data is mounted at /data (local mode) or downloaded (standalone mode)
# Output goes to /output
#
# All training parameters are passed as environment variables from
# the host wrapper script (train-wakeword.sh).

set -e

# =============================================================================
# Training Parameters (from environment, set by host wrapper)
# =============================================================================
WAKE_WORD="${WAKE_WORD:?ERROR: WAKE_WORD not set}"
MODEL_NAME="${MODEL_NAME:?ERROR: MODEL_NAME not set}"
N_SAMPLES="${N_SAMPLES:-50000}"
N_SAMPLES_VAL="${N_SAMPLES_VAL:-5000}"
AUGMENTATION_ROUNDS="${AUGMENTATION_ROUNDS:-2}"
TRAINING_STEPS="${TRAINING_STEPS:-100000}"
LAYER_SIZE="${LAYER_SIZE:-32}"

echo "=============================================="
echo "OpenWakeWord Docker Training"
echo "=============================================="
echo "Started: $(date)"
echo ""
echo "Wake word:           \"$WAKE_WORD\""
echo "Model name:          $MODEL_NAME"
echo ""
echo "Training Parameters:"
echo "  Samples:             $N_SAMPLES (val: $N_SAMPLES_VAL)"
echo "  Augmentation rounds: $AUGMENTATION_ROUNDS"
echo "  Training steps:      $TRAINING_STEPS"
echo "  Layer size:          $LAYER_SIZE"
echo ""

# GPU info
echo "[GPU Info]"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
python3 -c "import torch; print(f'  PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# =============================================================================
# Generate config YAML from parameters
# =============================================================================
WORKSPACE="/workspace"
cd "$WORKSPACE"
CONFIG="$WORKSPACE/${MODEL_NAME}_config.yml"

cat > "$CONFIG" << YAML
## Auto-generated configuration for "${WAKE_WORD}" wake word model

# Model name
model_name: "${MODEL_NAME}"

# Target phrase - what we want to detect
target_phrase:
  - "$(echo "$WAKE_WORD" | tr '[:upper:]' '[:lower:]')"

# Phrases to NOT activate on (auto-generated, reduces false positives)
custom_negative_phrases: []

# Number of synthetic samples to generate
n_samples: ${N_SAMPLES}
n_samples_val: ${N_SAMPLES_VAL}

# TTS batch size
tts_batch_size: 50

# Augmentation settings
augmentation_batch_size: 16
augmentation_rounds: ${AUGMENTATION_ROUNDS}

# Paths (relative to training directory)
piper_sample_generator_path: "./piper-sample-generator"
output_dir: "./${MODEL_NAME}_model"

# Room impulse responses
rir_paths:
  - "./mit_rirs"

# Background audio (MUSAN music from OpenSLR)
background_paths:
  - "./musan_music"

background_paths_duplication_rate:
  - 1

# Validation data
false_positive_validation_data_path: "./validation_set_features.npy"

# Pre-computed negative features
feature_data_files:
  "ACAV100M_sample": "./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"

# Batch composition
batch_n_per_class:
  "ACAV100M_sample": 1024
  "adversarial_negative": 100
  "positive": 100

# Model architecture
model_type: "dnn"
layer_size: ${LAYER_SIZE}

# Training parameters
steps: ${TRAINING_STEPS}
max_negative_weight: 1500
target_false_positives_per_hour: 0.1
target_accuracy: 0.7
target_recall: 0.5

# Learning rate
lr: 0.0001
YAML

echo "[Config] Generated $CONFIG"
echo ""

# Link openWakeWord and piper-sample-generator from container
ln -sf /app/openWakeWord "$WORKSPACE/openWakeWord"
ln -sf /app/piper-sample-generator "$WORKSPACE/piper-sample-generator"

# =============================================================================
# Data acquisition - standalone (download) or local (mount)
# =============================================================================
if [ "${STANDALONE:-0}" == "1" ]; then
    echo "[Standalone Mode] Downloading training data..."
    echo ""

    TARBALL_URL="${TARBALL_URL:-https://huggingface.co/datasets/brianckelley/atlas-voice-training-data/resolve/main/atlas-voice-training-data.tar.gz}"
    TARBALL_FILE="/tmp/atlas-voice-training-data.tar.gz"

    # -------------------------------------------------------------------------
    # Download tarball (~20GB)
    # -------------------------------------------------------------------------
    if [ -f "$WORKSPACE/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" ] && \
       [ -f "$WORKSPACE/validation_set_features.npy" ] && \
       [ -d "$WORKSPACE/musan_music" ]; then
        echo "  Training data already extracted. Skipping download."
    else
        echo "  Downloading training data tarball (~20GB)..."
        echo "  Source: $TARBALL_URL"
        echo ""
        wget --progress=bar:force:noscroll -O "$TARBALL_FILE" "$TARBALL_URL"
        echo ""
        echo "  Extracting tarball..."
        tar xzf "$TARBALL_FILE" -C "$WORKSPACE/"
        rm -f "$TARBALL_FILE"
        echo "  Extraction complete."
    fi
    echo ""

    # -------------------------------------------------------------------------
    # Download MIT Room Impulse Responses (~300MB)
    # Source: MIT directly (avoids HuggingFace rate limits)
    # Original files are 32kHz - must convert to 16kHz for training
    # -------------------------------------------------------------------------
    if [ -d "$WORKSPACE/mit_rirs" ] && [ -n "$(ls -A $WORKSPACE/mit_rirs/*.wav 2>/dev/null)" ]; then
        echo "  MIT RIRs already present. Skipping download."
    else
        echo "  Downloading MIT Room Impulse Responses (~300MB)..."
        echo "  Source: https://mcdermottlab.mit.edu"
        echo ""
        MIT_ZIP="/tmp/mit_rirs.zip"
        MIT_TEMP="/tmp/mit_rirs_raw"
        mkdir -p "$WORKSPACE/mit_rirs" "$MIT_TEMP"

        wget --progress=bar:force:noscroll -O "$MIT_ZIP" \
            "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
        echo ""

        echo "  Extracting..."
        unzip -q -o "$MIT_ZIP" -d "$MIT_TEMP"

        echo "  Converting to 16kHz mono (required for training)..."
        WAV_COUNT=0
        for f in $(find "$MIT_TEMP" -name "*.wav" -type f); do
            BASENAME=$(basename "$f")
            ffmpeg -y -loglevel error -i "$f" -ar 16000 -ac 1 "$WORKSPACE/mit_rirs/$BASENAME"
            WAV_COUNT=$((WAV_COUNT + 1))
        done

        rm -rf "$MIT_ZIP" "$MIT_TEMP"
        echo "  Converted $WAV_COUNT room impulse responses to 16kHz."
    fi
    echo ""

else
    echo "[Local Mode] Linking training data from /data mount..."

    # Link large files from /data mount
    for item in openwakeword_features_ACAV100M_2000_hrs_16bit.npy validation_set_features.npy musan_music; do
        if [ -e "/data/$item" ]; then
            ln -sf "/data/$item" "$WORKSPACE/$item"
            echo "  Linked: $item"
        else
            echo "  WARNING: Missing /data/$item"
        fi
    done

    # Handle MIT RIRs - check if in /data or need to download
    if [ -d "/data/mit_rirs" ] && [ -n "$(ls -A /data/mit_rirs/*.wav 2>/dev/null)" ]; then
        ln -sf /data/mit_rirs "$WORKSPACE/mit_rirs"
        echo "  Linked: mit_rirs"
    else
        echo "  MIT RIRs not found in /data. Downloading from MIT..."
        mkdir -p "$WORKSPACE/mit_rirs"
        MIT_ZIP="/tmp/mit_rirs.zip"
        MIT_TEMP="/tmp/mit_rirs_raw"
        mkdir -p "$MIT_TEMP"

        wget --progress=bar:force:noscroll -O "$MIT_ZIP" \
            "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"

        unzip -q -o "$MIT_ZIP" -d "$MIT_TEMP"

        WAV_COUNT=0
        for f in $(find "$MIT_TEMP" -name "*.wav" -type f); do
            BASENAME=$(basename "$f")
            ffmpeg -y -loglevel error -i "$f" -ar 16000 -ac 1 "$WORKSPACE/mit_rirs/$BASENAME"
            WAV_COUNT=$((WAV_COUNT + 1))
        done

        rm -rf "$MIT_ZIP" "$MIT_TEMP"
        echo "  Converted $WAV_COUNT room impulse responses to 16kHz."
    fi
fi

echo ""

# =============================================================================
# Verify all required files are present
# =============================================================================
echo "[Verify] Checking required files..."
MISSING=0
for item in openwakeword_features_ACAV100M_2000_hrs_16bit.npy validation_set_features.npy musan_music mit_rirs; do
    if [ -e "$WORKSPACE/$item" ]; then
        echo "  OK: $item"
    else
        echo "  MISSING: $item"
        MISSING=1
    fi
done

if [ "$MISSING" == "1" ]; then
    echo ""
    echo "ERROR: Required training data is missing. Cannot proceed."
    exit 1
fi

echo ""
echo "[Training] Starting..."
echo ""

# Step 6a: Generate synthetic speech clips
echo "  [6a] Generating synthetic speech clips..."
echo "       Started: $(date)"
python3 /app/openWakeWord/openwakeword/train.py --training_config "$CONFIG" --generate_clips
echo "       Finished: $(date)"
echo ""

# Step 6b: Augment clips (CPU only to avoid cuFFT conflicts)
echo "  [6b] Augmenting clips with noise/reverb..."
echo "       Started: $(date)"
CUDA_VISIBLE_DEVICES="" python3 /app/openWakeWord/openwakeword/train.py --training_config "$CONFIG" --augment_clips
echo "       Finished: $(date)"
echo ""

# Step 6c: Train neural network
echo "  [6c] Training neural network (GPU accelerated)..."
echo "       Started: $(date)"
python3 /app/openWakeWord/openwakeword/train.py --training_config "$CONFIG" --train_model || true
echo "       Finished: $(date)"

# Check if model was saved (segfault on cleanup is expected - model is saved before that)
MODEL_FILE="${MODEL_NAME}_model/${MODEL_NAME}.onnx"
if [ -f "$MODEL_FILE" ]; then
    echo "  Training complete. Model saved: $MODEL_FILE"
else
    echo "  ERROR: Training failed - model file not found"
    exit 1
fi

# Convert ONNX to TFLite
TFLITE_FILE="${MODEL_NAME}_model/${MODEL_NAME}.tflite"
echo "  Converting ONNX to TFLite..."
python3 << EOF
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load("${MODEL_FILE}")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("${MODEL_NAME}_model/${MODEL_NAME}_tf")

converter = tf.lite.TFLiteConverter.from_saved_model("${MODEL_NAME}_model/${MODEL_NAME}_tf")
tflite_model = converter.convert()

with open("${TFLITE_FILE}", "wb") as f:
    f.write(tflite_model)

print(f"  TFLite model saved: ${TFLITE_FILE}")
EOF

# Clean up intermediate TF model
rm -rf "${MODEL_NAME}_model/${MODEL_NAME}_tf"

# Copy output to mounted output directory
echo ""
echo "[Output] Copying model files to /output..."
cp -v "${MODEL_NAME}_model/${MODEL_NAME}.onnx" /output/
cp -v "${MODEL_NAME}_model/${MODEL_NAME}.tflite" /output/ 2>/dev/null || echo "  TFLite not available"

echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "=============================================="
echo "Finished: $(date)"
echo ""
echo "Model files in /output:"
ls -la /output/
echo ""
