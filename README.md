# Blink

A touchless vision controller for Linux workstations. Blink uses a standard webcam to detect hand gestures, track fingertip movement as an air mouse and register eye blinks. It translates these visual cues into native keyboard shortcuts, mouse clicks, scrolls and shell commands in real time.

A local web interface provides a live video feed with an augmented reality HUD, interactive sensitivity controls, gesture logs and configuration management. All vision processing runs entirely on your local machine using MediaPipe and OpenCV.

---

## Features

### Dual-Hand Spatial Gestures
- **Window Expand**: Spread both hands apart horizontally to maximize or expand the active window (`super+Up`).
- **Window Shrink**: Bring both hands together horizontally to shrink or restore the active window (`super+Down`).
- **Workspace Throws**: Sweep both hands to the left or right in unison to jump between virtual desktops (`ctrl+alt+Left` and `ctrl+alt+Right`).

### Single-Hand Gestures
- **Palm Push**: Push a flat open palm forward toward the camera to minimize everything and show the desktop (`super+d`).
- **Palm to Fist**: Transition smoothly from an open palm to a closed fist to close the active window (`super+h`).
- **Fist to Palm**: Transition from a closed fist to an open palm to switch back to the previous window (`alt+Tab`).
- **Horizontal Swipes**: Quick horizontal flick left or right to switch desktop workspaces.
- **Continuous Scroll**: Move your hand vertically above or below the deadzone to scroll documents and web pages smoothly.
- **Double Clap**: Clap your hands twice within a short window to open your default web browser.
- **Finger to Lips**: Hold your index finger against your lips for 350ms to toggle system audio mute (`XF86AudioMute`).
- **Thumb Signals**: Thumbs up increases volume, thumbs down decreases volume.

### Air Mouse Fingertip Tracking
- **Activation**: Hold your index finger steady for two seconds to lock onto the cursor.
- **Adaptive Smoothing**: Multi-tier exponential moving average filters out trembling during micro-aiming while remaining responsive during rapid hand movements.
- **Deadband Jitter Filter**: Sub-3px movements are ignored so the cursor stays perfectly still when reading or hovering over small buttons.
- **Instant Fingertip Click**: Bend the tip of your index finger downward to register a left click immediately. The cursor coordinates freeze for 150ms during the click to prevent target slipping.
- **Forward Tap Click**: Poke your pointing finger forward toward the camera as an alternative left click method.
- **Middle Finger Right Click**: Flick your middle finger outward while pointing to trigger a context menu right click.
- **Disengage**: Clench your hand into a fist to exit air mouse mode.

### Optical Eye Blink Mode
- Toggle from hand gestures to facial eye tracking through the dashboard.
- Uses eye aspect ratio calculations and temporal thresholding to detect double blinks, prolonged blinks, left winks and right winks.

### Privacy and Local Processing
- **Face Blur Shield**: Optional one-click toggle in the header blurs and pixelates your face in the video feed.
- **Zero External Network Calls**: MediaPipe inference runs on your CPU using local model files. No image frames leave your computer.

---

## Hardware and System Requirements

- **Operating System**: Linux (Ubuntu, Debian, Fedora or Arch). Tested on X11 and Wayland. For synthetic mouse and keyboard dispatch, `pynput` or `xdotool` is used.
- **Camera**: Any USB webcam or integrated laptop camera supporting 640x480 resolution at 30 FPS.
- **Python**: Python 3.10 or newer (Python 3.12 recommended).
- **Package Tool**: `uv` (recommended) or standard `pip`.

---

## Quick Start

### 1. Run with Helper Script

The included `run.sh` script creates a virtual environment, downloads required MediaPipe task models if they are missing and starts the application:

```bash
chmod +x run.sh
./run.sh
```

### 2. Manual Installation

If you prefer setting up manually with standard tools:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe numpy fastapi uvicorn websockets pynput httpx

# Download models if not already present
curl -s -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -s -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

# Start the application
python main.py
```

### 3. Open the Dashboard

Open your web browser and navigate to:

```
http://127.0.0.1:8000
```

The control center displays your camera stream, detection HUD, FPS counter, latency tracker and real-time gesture logs.

---

## Gesture and Action Mapping Reference

| Gesture | Movement or Trigger | Default Action | Hotkey / Command |
| :--- | :--- | :--- | :--- |
| Window Expand | Spread both hands apart | Maximize window | `super+Up` |
| Window Shrink | Bring both hands together | Restore window | `super+Down` |
| Dual Swipe Left | Sweep both hands left | Switch desktop left | `ctrl+alt+Left` |
| Dual Swipe Right | Sweep both hands right | Switch desktop right | `ctrl+alt+Right` |
| Palm Push | Thrust flat open palm forward | Show desktop | `super+d` |
| Palm to Fist | Open hand into closed fist | Close window | `super+h` |
| Fist to Palm | Closed fist into open hand | Previous window | `alt+Tab` |
| Air Click | Bend index fingertip downward | Left click / Open terminal | Pointer Click |
| Forward Tap | Poke index finger forward | Left click | Pointer Click |
| Middle Flick | Flick middle finger out | Right click | Pointer Right Click |
| Steady Finger | Hold index finger still for 2s | Engage air mouse | Mouse Control |
| Fist in Mouse Mode | Clench fist while tracking | Exit air mouse | Release Mouse |
| Double Clap | Clap hands twice rapidly | Open browser | `xdg-open https://google.com` |
| Finger to Lips | Finger on lips for 350ms | Toggle mute | `XF86AudioMute` |
| Swipe Left | Fast hand flick left | Desktop left | `ctrl+alt+Left` |
| Swipe Right | Fast hand flick right | Desktop right | `ctrl+alt+Right` |
| Hand Up | Move hand upward | Scroll up | Continuous scroll |
| Hand Down | Move hand downward | Scroll down | Continuous scroll |
| Thumbs Up | Thumb held upward | Volume up | `XF86AudioRaiseVolume` |
| Thumbs Down | Thumb held downward | Volume down | `XF86AudioLowerVolume` |

---

## Configuration and Presets

Configuration is saved in `config.json` and can be edited directly or modified through the web interface.

### Built-in Presets
- **Browsing**: Optimized for tab navigation, scrolling, window resizing and links.
- **Media**: Configured for volume adjustments, playback toggling, muting and track skipping.
- **Presentation**: Tailored for slide progression, full-screen toggles and presentation flow.

### Keybinding Format
Actions support three execution formats:
- **Hotkey**: Key combinations such as `super+d`, `ctrl+alt+Left` or `alt+Tab`.
- **Command**: Shell commands executed via subprocess, for example `gnome-terminal` or `xdg-open https://google.com`.
- **Native Methods**: Built-in actions such as `scroll_up`, `scroll_down`, `minimize_window` and `restore_window`.

Cooldown delays, gesture sensitivity thresholds and air mouse smoothing weights can all be fine-tuned in `config.json` without modifying any Python code.

---

## Testing

Run the automated test suite to verify vision engines, action dispatchers and web endpoints:

```bash
.venv/bin/python -m unittest discover tests/
```

All 46 unit tests validate:
- One-handed gesture classification and debounce timers
- Dual-hand distance tracking and velocity thresholds
- Palm-to-fist transition isolation and suppression logic
- Air mouse deadband, smoothing and fingertip bend detection
- Facial landmarker wink and blink calculations
- Action dispatcher hotkey, command and native execution pipelines
- REST API configuration, status and preset endpoints

---

## Architecture Overview

```
blink/
├── main.py                     # Application entry point and server startup
├── run.sh                      # Shell bootstrap and environment manager
├── config.json                 # User configuration and gesture mappings
├── face_landmarker.task        # MediaPipe facial blendshape model
├── gesture_recognizer.task     # MediaPipe hand gesture model
├── blink_core/
│   ├── action_dispatcher.py    # Native keypress, mouse and shell dispatcher
│   ├── camera.py               # OpenCV capture thread and frame streamer
│   ├── config_manager.py       # Configuration loader, presets and persistence
│   ├── detector.py             # MediaPipe face landmarker and EAR calculator
│   ├── gesture_engine.py       # Blink and wink temporal state machine
│   ├── hand_detector.py        # MediaPipe hand tracking and HUD renderer
│   └── hand_gesture_engine.py  # Spatial hand gestures and air mouse tracking
├── server/
│   └── app.py                  # FastAPI web server, WebSocket and REST API
├── static/
│   ├── index.html              # Web dashboard interface
│   ├── css/style.css           # Dashboard styling
│   └── js/app.js               # Frontend telemetry, HUD and event handling
└── tests/
    ├── test_blinkflow.py       # Blink engine and API integration tests
    └── test_hand_flow.py       # Hand gesture and spatial tests
```

---

## License

MIT License.
