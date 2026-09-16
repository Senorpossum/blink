#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "[BlinkFlow] Virtual environment not found. Creating with uv..."
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python opencv-python mediapipe numpy fastapi uvicorn websockets pynput
fi

# Ensure models exist
if [ ! -f "face_landmarker.task" ]; then
    echo "[BlinkFlow] Downloading MediaPipe Face Landmarker model..."
    curl -s -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
fi

if [ ! -f "gesture_recognizer.task" ]; then
    echo "[BlinkFlow] Downloading MediaPipe Gesture Recognizer model..."
    curl -s -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
fi

echo "[BlinkFlow] Launching BlinkFlow Application..."
exec .venv/bin/python main.py "$@"
