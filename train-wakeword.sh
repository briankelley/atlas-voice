#!/bin/bash
# Wrapper script to build and run atlas-voice training in Docker
#
# Usage: ./train-wakeword.sh [--rebuild] [--standalone]
#
# Options:
#   --rebuild      Force rebuild of Docker image even if it exists
#   --standalone   Run without local training data (downloads ~20GB from remote)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="atlas-voice-training"
CONTAINER_NAME="atlas-training-$(date +%Y%m%d-%H%M%S)"

# Training data location (where the big .npy files are)
DATA_DIR="/media/brian/DATA/code/tools/speech-to-text/atlas/training"

# Output directory for trained models
OUTPUT_DIR="$SCRIPT_DIR/docker-output"
mkdir -p "$OUTPUT_DIR"

# Tarball download URL for standalone mode
TARBALL_URL="${TARBALL_URL:-https://huggingface.co/datasets/brianckelley/atlas-voice-training-data/resolve/main/atlas-voice-training-data.tar.gz}"

# Parse arguments
REBUILD=false
STANDALONE=false
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=true ;;
        --standalone) STANDALONE=true ;;
    esac
done

# Auto-detect standalone mode if local data doesn't exist
if [ ! -d "$DATA_DIR" ] || [ ! -f "$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" ]; then
    STANDALONE=true
fi

# =========================================================================
# Pre-flight checks
# =========================================================================

# Check for NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. GPU required for training."
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia\|Default Runtime.*nvidia"; then
    if ! dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
        echo "WARNING: nvidia-container-toolkit may not be installed."
        echo "  Install with: sudo apt install nvidia-container-toolkit"
        echo "  Then restart Docker: sudo systemctl restart docker"
        echo ""
        read -p "  Try anyway? [y/N] " -r
        if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
fi

# Check for required files in local mode
if [ "$STANDALONE" == "false" ]; then
    for f in openwakeword_features_ACAV100M_2000_hrs_16bit.npy validation_set_features.npy; do
        if [ ! -f "$DATA_DIR/$f" ]; then
            echo "ERROR: Required file not found: $DATA_DIR/$f"
            exit 1
        fi
    done
fi

# =========================================================================
# Build Docker image
# =========================================================================
if ! docker image inspect "$IMAGE_NAME" &>/dev/null || [ "$REBUILD" == "true" ]; then
    echo "Building Docker image (this takes a few minutes the first time)..."
    echo ""
    cd "$SCRIPT_DIR"
    docker build -f Dockerfile.training -t "$IMAGE_NAME" .
    echo ""
    echo "Docker image built successfully."
    echo ""
else
    echo "Using existing Docker image: $IMAGE_NAME"
    echo ""
fi

# =========================================================================
# User interaction - proceed, wake word, training settings
# =========================================================================

echo "=============================================="
echo "  OpenWakeWord Custom Wake Word Training"
echo "=============================================="
echo ""
echo "The training environment is ready. Next we need"
echo "to download ~20 GB of training data to train a"
echo "custom wake word model."
echo ""

read -p "Do you want to proceed? [y/N] " -r
echo ""
if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "To download training data and train a new word"
    echo "later, just rerun this script (./train-wakeword.sh)"
    exit 0
fi

# --- Wake word ---
echo "----------------------------------------------"
echo "  Wake Word Configuration"
echo "----------------------------------------------"
echo ""
echo "What wake word do you want to train?"
echo ""
echo "Recommended: Two-word phrases work best. \"Hey Atlas\""
echo "consistently outperformed \"Atlas\" by 10+ points in"
echo "accuracy and 18+ points in recall (not having to"
echo "repeat yourself) in tested configurations."
echo ""
echo "Examples: \"Hey Atlas\", \"Hey Jarvis\", \"Okay Computer\""
echo ""
read -p "Wake word: " WAKE_WORD
echo ""

if [ -z "$WAKE_WORD" ]; then
    echo "ERROR: Wake word cannot be empty."
    exit 1
fi

# Derive model name from wake word (lowercase, spaces to underscores)
MODEL_NAME=$(echo "$WAKE_WORD" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

# --- Training settings ---
# Defaults (Jan 30 model - best balance of accuracy and recall)
N_SAMPLES=50000
N_SAMPLES_VAL=5000
AUGMENTATION_ROUNDS=2
TRAINING_STEPS=100000
LAYER_SIZE=32

echo "----------------------------------------------"
echo "  Training Settings"
echo "----------------------------------------------"
echo ""
echo "These are the default settings that will be used"
echo "unless you decide to change them. They produce the"
echo "best balance of accuracy and recall."
echo ""
echo "  Training samples:     $N_SAMPLES"
echo "  Augmentation rounds:  $AUGMENTATION_ROUNDS"
echo "  Training steps:       $TRAINING_STEPS"
echo "  Layer size (neurons): $LAYER_SIZE"
echo ""
read -p "Use recommended settings? (select \"n\" for descriptions) [Y/n] " -r
echo ""

if [[ "$REPLY" =~ ^[Nn]$ ]]; then
    # --- Training samples ---
    echo "----------------------------------------------"
    echo "  Training Samples              default: 50000"
    echo "----------------------------------------------"
    echo "How many synthetic speech clips to generate"
    echo "for training."
    echo ""
    echo "More samples = more pronunciation variety."
    echo "Fewer samples = faster training."
    echo ""
    echo "In testing, doubling from 50k to 100k did NOT"
    echo "improve accuracy or recall. It reduced false"
    echo "positives but at the cost of missing more real"
    echo "wake words."
    echo ""
    read -p "Samples [default: $N_SAMPLES]: " INPUT
    N_SAMPLES="${INPUT:-$N_SAMPLES}"
    echo ""

    # --- Augmentation rounds ---
    echo "----------------------------------------------"
    echo "  Augmentation Rounds              default: 2"
    echo "----------------------------------------------"
    echo "How many times each clip is re-processed with"
    echo "different background noise, reverb, and room"
    echo "conditions."
    echo ""
    echo "More rounds = better noise/environment handling."
    echo "Fewer rounds = faster training."
    echo ""
    echo "In testing, 3 rounds produced no measurable"
    echo "improvement over 2 for wake word detection."
    echo "May help if you plan to use this where there's"
    echo "plenty of ambient noise."
    echo ""
    read -p "Augmentation rounds [default: $AUGMENTATION_ROUNDS]: " INPUT
    AUGMENTATION_ROUNDS="${INPUT:-$AUGMENTATION_ROUNDS}"
    echo ""

    # --- Training steps ---
    echo "----------------------------------------------"
    echo "  Training Steps                default: 100000"
    echo "----------------------------------------------"
    echo "How many steps the neural network trains."
    echo "More steps gives the model more time to learn,"
    echo "but with diminishing returns."
    echo ""
    echo "This is the most GPU-intensive phase."
    echo "100k steps on an RTX 4090 takes ~20 minutes."
    echo "150k steps did not improve results in testing."
    echo ""
    read -p "Training steps [default: $TRAINING_STEPS]: " INPUT
    TRAINING_STEPS="${INPUT:-$TRAINING_STEPS}"
    echo ""

    # --- Layer size ---
    echo "----------------------------------------------"
    echo "  Layer Size (Neurons)             default: 32"
    echo "----------------------------------------------"
    echo "The number of neurons in each hidden layer."
    echo "More neurons = more capacity to represent"
    echo "subtle differences in pronunciation."
    echo ""
    echo "In testing, 64 neurons produced statistically"
    echo "identical results to 32 for wake word models."
    echo "The output model stays tiny either way (~200 KB)."
    echo ""
    read -p "Layer size [default: $LAYER_SIZE]: " INPUT
    LAYER_SIZE="${INPUT:-$LAYER_SIZE}"
    echo ""
fi

# Validation samples = 10% of training samples
N_SAMPLES_VAL=$((N_SAMPLES / 10))

# --- Confirm ---
echo "----------------------------------------------"
echo "  Ready to Train"
echo "----------------------------------------------"
echo ""
echo "  Wake word:           \"$WAKE_WORD\""
echo "  Samples:             $N_SAMPLES"
echo "  Augmentation rounds: $AUGMENTATION_ROUNDS"
echo "  Training steps:      $TRAINING_STEPS"
echo "  Layer size:          $LAYER_SIZE"
echo ""
read -p "Start training? [Y/n] " -r
echo ""
if [[ "$REPLY" =~ ^[Nn]$ ]]; then
    echo "Aborted. Rerun ./train-wakeword.sh when ready."
    exit 0
fi

# =========================================================================
# Launch container
# =========================================================================
echo "Starting training container..."
echo "Container name: $CONTAINER_NAME"
echo ""
echo "=============================================="
echo ""

DOCKER_ARGS=(
    --gpus all
    --shm-size=32g
    --name "$CONTAINER_NAME"
    --rm
    -v "$OUTPUT_DIR:/output:rw"
    -e WAKE_WORD="$WAKE_WORD"
    -e MODEL_NAME="$MODEL_NAME"
    -e N_SAMPLES="$N_SAMPLES"
    -e N_SAMPLES_VAL="$N_SAMPLES_VAL"
    -e AUGMENTATION_ROUNDS="$AUGMENTATION_ROUNDS"
    -e TRAINING_STEPS="$TRAINING_STEPS"
    -e LAYER_SIZE="$LAYER_SIZE"
)

if [ "$STANDALONE" == "true" ]; then
    DOCKER_ARGS+=( -e STANDALONE=1 )
    DOCKER_ARGS+=( -e TARBALL_URL="$TARBALL_URL" )
else
    DOCKER_ARGS+=( -v "$DATA_DIR:/data:ro" )
fi

docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME"

echo ""
echo "=============================================="
echo "Training complete!"
echo ""
echo "Model files are in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
echo ""
echo "To use the model:"
echo "  cp $OUTPUT_DIR/${MODEL_NAME}.tflite ~/.local/share/openwakeword/"
echo "=============================================="
