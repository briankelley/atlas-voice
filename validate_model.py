#!/usr/bin/env python3
"""
Validate a wake word model against test features.
Usage: python validate_model.py /path/to/model.tflite
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Paths to test features - set ATLAS_FEATURES_DIR to your training output directory
# Example: export ATLAS_FEATURES_DIR=/path/to/training/hey_atlas_model/hey_atlas
FEATURES_DIR = os.environ.get("ATLAS_FEATURES_DIR", os.path.join(os.path.dirname(__file__), "docker-output"))
POSITIVE_TEST = os.path.join(FEATURES_DIR, "positive_features_test.npy")
NEGATIVE_TEST = os.path.join(FEATURES_DIR, "negative_features_test.npy")

# Validation negative features (hours of random audio)
VALIDATION_FEATURES = os.environ.get("ATLAS_VALIDATION_FEATURES", os.path.join(FEATURES_DIR, "validation_set_features.npy"))

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details[0]['index'], output_details[0]['index']

def predict(interpreter, input_index, output_index, features):
    predictions = []
    for i in range(features.shape[0]):
        interpreter.set_tensor(input_index, features[i:i+1].astype(np.float32))
        interpreter.invoke()
        pred = interpreter.get_tensor(output_index)[0][0]
        predictions.append(pred)
    return np.array(predictions)

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_model.py /path/to/model.tflite [threshold]")
        sys.exit(1)

    model_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    print(f"Loading model: {model_path}")
    print(f"Threshold: {threshold}")
    print()

    interpreter, input_idx, output_idx = load_tflite_model(model_path)

    # Load test features
    print("Loading test features...")
    pos_features = np.load(POSITIVE_TEST)
    neg_features = np.load(NEGATIVE_TEST)

    print(f"  Positive samples: {pos_features.shape[0]}")
    print(f"  Negative samples: {neg_features.shape[0]}")
    print()

    # Run predictions
    print("Running predictions on positive samples...")
    pos_preds = predict(interpreter, input_idx, output_idx, pos_features)

    print("Running predictions on negative samples...")
    neg_preds = predict(interpreter, input_idx, output_idx, neg_features)

    # Calculate metrics
    true_positives = np.sum(pos_preds >= threshold)
    false_negatives = np.sum(pos_preds < threshold)
    true_negatives = np.sum(neg_preds < threshold)
    false_positives = np.sum(neg_preds >= threshold)

    total = len(pos_preds) + len(neg_preds)
    accuracy = (true_positives + true_negatives) / total
    recall = true_positives / len(pos_preds) if len(pos_preds) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # False positives per hour (assuming ~12 predictions per second for streaming audio)
    # neg_features represents some amount of audio - estimate based on feature count
    neg_hours = len(neg_preds) / (12 * 3600)  # rough estimate
    fp_per_hour = false_positives / neg_hours if neg_hours > 0 else 0

    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Recall:                {recall:.4f} ({recall*100:.2f}%)")
    print(f"Precision:             {precision:.4f} ({precision*100:.2f}%)")
    print(f"True Positives:        {true_positives}/{len(pos_preds)}")
    print(f"False Negatives:       {false_negatives}/{len(pos_preds)}")
    print(f"True Negatives:        {true_negatives}/{len(neg_preds)}")
    print(f"False Positives:       {false_positives}/{len(neg_preds)}")
    print(f"Est. FP/hour:          {fp_per_hour:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
