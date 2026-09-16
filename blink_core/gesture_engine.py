"""
Gesture Engine for BlinkFlow.
Distinguishes between natural blinks, double blinks, triple blinks, held blinks, and winks.
"""

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .config_manager import SensitivityConfig

logger = logging.getLogger("GestureEngine")


class EyeState(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class FaceGesture(Enum):
    NONE = "NONE"
    SINGLE_BLINK = "SINGLE_BLINK"
    DOUBLE_BLINK = "DOUBLE_BLINK"
    TRIPLE_BLINK = "TRIPLE_BLINK"
    LONG_BLINK = "LONG_BLINK"
    LEFT_WINK = "LEFT_WINK"
    RIGHT_WINK = "RIGHT_WINK"
    LONG_LEFT_WINK = "LONG_LEFT_WINK"
    LONG_RIGHT_WINK = "LONG_RIGHT_WINK"


class GestureEngine:
    def __init__(self, sensitivity: SensitivityConfig):
        self.sensitivity = sensitivity
        self.on_gesture_detected: Optional[Callable[[FaceGesture, float], None]] = None

        # Tracking state
        self.both_closed = False
        self.left_wink = False
        self.right_wink = False

        # Timestamps (seconds)
        self.both_close_start: Optional[float] = None
        self.left_wink_start: Optional[float] = None
        self.right_wink_start: Optional[float] = None

        # Multi-blink counting
        self.blink_count = 0
        self.last_blink_release_time: float = 0.0

        # Long blink trigger flags (to avoid re-triggering while still held)
        self.long_blink_triggered = False
        self.long_left_wink_triggered = False
        self.long_right_wink_triggered = False

        # Cooldown
        self.last_trigger_time: float = 0.0

        # State description for HUD telemetry
        self.current_state_text = "IDLE"

    def update_sensitivity(self, sensitivity: SensitivityConfig) -> None:
        self.sensitivity = sensitivity

    def process(self, left_score: float, right_score: float, face_detected: bool) -> Optional[FaceGesture]:
        now = time.time()

        if not face_detected:
            self._reset_transient_state()
            self.current_state_text = "NO_FACE"
            return None

        # Check cooldown
        cooldown_sec = self.sensitivity.cooldown_ms / 1000.0
        if (now - self.last_trigger_time) < cooldown_sec:
            remaining = int((cooldown_sec - (now - self.last_trigger_time)) * 1000)
            self.current_state_text = f"COOLDOWN ({remaining}ms)"
            return None

        # Evaluate eye openness
        thresh = self.sensitivity.blink_threshold
        wink_open_thresh = self.sensitivity.wink_open_threshold

        is_left_closed = left_score >= thresh
        is_right_closed = right_score >= thresh

        is_left_open = left_score < wink_open_thresh
        is_right_open = right_score < wink_open_thresh

        detected_gesture: Optional[FaceGesture] = None

        # -------------------------------------------------------------
        # 1. BOTH EYES CLOSED
        # -------------------------------------------------------------
        if is_left_closed and is_right_closed:
            # If we were previously winking, reset wink states
            self.left_wink_start = None
            self.right_wink_start = None

            if not self.both_closed:
                # Started closing
                self.both_closed = True
                self.both_close_start = now
                self.long_blink_triggered = False
            else:
                # Holding closed
                hold_duration_ms = (now - self.both_close_start) * 1000.0
                if (hold_duration_ms >= self.sensitivity.long_blink_duration_ms) and not self.long_blink_triggered:
                    # Trigger LONG_BLINK immediately upon reaching hold threshold
                    detected_gesture = FaceGesture.LONG_BLINK
                    self.long_blink_triggered = True
                    self.current_state_text = "LONG BLINK TRIGGERED"
                    self._fire_gesture(FaceGesture.LONG_BLINK, hold_duration_ms)
                    return detected_gesture
                elif not self.long_blink_triggered:
                    self.current_state_text = f"HOLDING ({int(hold_duration_ms)}ms)"

        # -------------------------------------------------------------
        # 2. BOTH EYES OPEN (or released from closed)
        # -------------------------------------------------------------
        elif is_left_open and is_right_open:
            if self.both_closed:
                # Eyes just opened!
                self.both_closed = False
                close_duration_ms = (now - self.both_close_start) * 1000.0 if self.both_close_start else 0.0

                if self.long_blink_triggered:
                    # Already handled as a long blink, just reset
                    self.both_close_start = None
                    self.blink_count = 0
                elif close_duration_ms >= self.sensitivity.min_blink_duration_ms:
                    # Valid short/intentional blink!
                    self.blink_count += 1
                    self.last_blink_release_time = now
                    self.current_state_text = f"BLINK x{self.blink_count}"
                else:
                    # Too short, glitch
                    self.current_state_text = "GLITCH / FLUTTER"

            # Check for left wink release
            if self.left_wink_start is not None:
                duration_ms = (now - self.left_wink_start) * 1000.0
                if not self.long_left_wink_triggered and (duration_ms >= self.sensitivity.min_blink_duration_ms):
                    detected_gesture = FaceGesture.LEFT_WINK
                    self._fire_gesture(FaceGesture.LEFT_WINK, duration_ms)
                self.left_wink_start = None
                self.long_left_wink_triggered = False

            # Check for right wink release
            if self.right_wink_start is not None:
                duration_ms = (now - self.right_wink_start) * 1000.0
                if not self.long_right_wink_triggered and (duration_ms >= self.sensitivity.min_blink_duration_ms):
                    detected_gesture = FaceGesture.RIGHT_WINK
                    self._fire_gesture(FaceGesture.RIGHT_WINK, duration_ms)
                self.right_wink_start = None
                self.long_right_wink_triggered = False

            # Check multi-blink expiration or completion
            if self.blink_count > 0:
                elapsed_since_last_release = (now - self.last_blink_release_time) * 1000.0

                if self.blink_count == 3:
                    detected_gesture = FaceGesture.TRIPLE_BLINK
                    self._fire_gesture(FaceGesture.TRIPLE_BLINK, elapsed_since_last_release)
                    self.blink_count = 0
                elif self.blink_count == 2:
                    if elapsed_since_last_release > 120:
                        # Slight settle time to allow a third blink if fast, or fire double blink
                        if elapsed_since_last_release >= self.sensitivity.double_blink_window_ms:
                            detected_gesture = FaceGesture.DOUBLE_BLINK
                            self._fire_gesture(FaceGesture.DOUBLE_BLINK, elapsed_since_last_release)
                            self.blink_count = 0
                elif self.blink_count == 1:
                    if elapsed_since_last_release > self.sensitivity.double_blink_window_ms:
                        # Double blink window expired without second blink
                        # This was a single blink (either natural or intentional)
                        detected_gesture = FaceGesture.SINGLE_BLINK
                        # Only fire callback if single blink is enabled in mappings
                        self._fire_gesture(FaceGesture.SINGLE_BLINK, elapsed_since_last_release)
                        self.blink_count = 0
                        self.current_state_text = "IDLE"
                    else:
                        self.current_state_text = f"WAITING BLINK 2 ({int(self.sensitivity.double_blink_window_ms - elapsed_since_last_release)}ms)"

        # -------------------------------------------------------------
        # 3. LEFT WINK (Left closed, Right open)
        # -------------------------------------------------------------
        elif is_left_closed and is_right_open:
            self.both_closed = False
            self.right_wink_start = None

            if self.left_wink_start is None:
                self.left_wink_start = now
                self.long_left_wink_triggered = False
            else:
                hold_ms = (now - self.left_wink_start) * 1000.0
                if (hold_ms >= self.sensitivity.long_blink_duration_ms) and not self.long_left_wink_triggered:
                    detected_gesture = FaceGesture.LONG_LEFT_WINK
                    self.long_left_wink_triggered = True
                    self._fire_gesture(FaceGesture.LONG_LEFT_WINK, hold_ms)
                elif not self.long_left_wink_triggered:
                    self.current_state_text = f"LEFT WINK ({int(hold_ms)}ms)"

        # -------------------------------------------------------------
        # 4. RIGHT WINK (Right closed, Left open)
        # -------------------------------------------------------------
        elif is_right_closed and is_left_open:
            self.both_closed = False
            self.left_wink_start = None

            if self.right_wink_start is None:
                self.right_wink_start = now
                self.long_right_wink_triggered = False
            else:
                hold_ms = (now - self.right_wink_start) * 1000.0
                if (hold_ms >= self.sensitivity.long_blink_duration_ms) and not self.long_right_wink_triggered:
                    detected_gesture = FaceGesture.LONG_RIGHT_WINK
                    self.long_right_wink_triggered = True
                    self._fire_gesture(FaceGesture.LONG_RIGHT_WINK, hold_ms)
                elif not self.long_right_wink_triggered:
                    self.current_state_text = f"RIGHT WINK ({int(hold_ms)}ms)"

        return detected_gesture

    def _fire_gesture(self, gesture: FaceGesture, duration_ms: float) -> None:
        self.last_trigger_time = time.time()
        self.current_state_text = f"TRIGGER: {gesture.value}"
        logger.info(f"Gesture recognized: {gesture.value} ({duration_ms:.1f}ms)")
        if self.on_gesture_detected:
            try:
                self.on_gesture_detected(gesture, duration_ms)
            except Exception as e:
                logger.error(f"Error in on_gesture_detected handler: {e}")

    def _reset_transient_state(self) -> None:
        self.both_closed = False
        self.both_close_start = None
        self.left_wink_start = None
        self.right_wink_start = None
        self.blink_count = 0
        self.long_blink_triggered = False
        self.long_left_wink_triggered = False
        self.long_right_wink_triggered = False
