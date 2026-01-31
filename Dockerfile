# OpenWakeWord Custom Model Training Environment
#
# Build:   docker build -t atlas-voice-training .
# Run:     docker run --gpus all -v $(pwd)/output:/output atlas-voice-training
#
# Note: GPU passthrough requires nvidia-docker2 for TensorFlow training step

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    espeak-ng \
    libespeak-ng-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy training files
COPY train.sh hey_atlas_config.yml ./
RUN chmod +x train.sh

# Create output directory
RUN mkdir -p /output

# Default command - run training and copy output
CMD ["bash", "-c", "./train.sh && cp -r *_model/*.tflite *_model/*.onnx /output/ 2>/dev/null || echo 'No model files to copy'"]
