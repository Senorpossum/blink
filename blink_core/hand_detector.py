"""
Hand Landmarker and Gesture Detector using MediaPipe GestureRecognizer.
Detects 21 3D hand joints, pinch distances, motion vectors, and static poses.
"""

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


@dataclass
class HandData:
    hand_index: int = 0
    handedness: str = "Right"  # "Left" or "Right"
    # Static classified gesture (e.g. "Open_Palm", "Closed_Fist", "Pointing_Up", "Victory", "Thumb_Up", "Thumb_Down", "None")
    gesture_name: str = "None"
    gesture_score: float = 0.0
    # Normalized pinch distance: ~0.15 when touching, >0.8 when wide open
    pinch_distance: float = 1.0
    is_pinched: bool = False
    # Palm centroid (normalized 0.0 - 1.0)
    center_x: float = 0.5
    center_y: float = 0.5
    # Key fingertips (x, y)
    thumb_tip: Tuple[float, float] = (0.0, 0.0)
    index_tip: Tuple[float, float] = (0.0, 0.0)
    middle_tip: Tuple[float, float] = (0.0, 0.0)
    wrist: Tuple[float, float] = (0.0, 0.0)
    # 3D Depth and relative metrics for forward tap and flick detection
    index_rel_z: float = 0.0  # L8.z - L5.z (forward tap poke depth relative to index knuckle)
    middle_dist: float = 0.0  # distance(L12, L9) (middle finger extension distance)
    index_bend_ratio: float = 1.0  # dist(8,6) / (dist(8,7) + dist(7,6)) - dips < 0.78 when clicking
    index_curl_ratio: float = 1.0  # dist(8,5) / total length - curls < 0.76 when clicking
    # Full 21 landmarks (x, y, z)
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class HandDetectionResult:
    hands_detected: bool = False
    num_hands: int = 0
    hands: List[HandData] = field(default_factory=list)
    primary_hand: Optional[HandData] = None
    latency_ms: float = 0.0


class HandDetector:
    # 21 Connections for hand skeleton rendering
    HAND_CONNECTIONS = [
        # Palm
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),# Ring
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17)                               # Palm base
    ]

    def __init__(self, model_path: str = "gesture_recognizer.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe Gesture Recognizer model not found at {model_path}")

        self.model_path = model_path
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray, draw_overlay: bool = True, extra_info: Optional[Dict[str, Any]] = None) -> Tuple[HandDetectionResult, np.ndarray]:
        t0 = time.time()
        h, w, _ = frame_bgr.shape
        res = HandDetectionResult()

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection = self.recognizer.recognize(mp_image)
        latency = (time.time() - t0) * 1000.0
        res.latency_ms = round(latency, 1)

        annotated_frame = frame_bgr.copy() if draw_overlay else frame_bgr

        if not detection.hand_landmarks or len(detection.hand_landmarks) == 0:
            res.hands_detected = False
            return res, annotated_frame

        res.hands_detected = True
        res.num_hands = len(detection.hand_landmarks)

        for i, landmarks in enumerate(detection.hand_landmarks):
            h_data = HandData(hand_index=i)

            # Handedness
            if detection.handedness and len(detection.handedness) > i and detection.handedness[i]:
                h_data.handedness = detection.handedness[i][0].display_name or detection.handedness[i][0].category_name

            # Gestures
            if detection.gestures and len(detection.gestures) > i and detection.gestures[i]:
                top_gesture = detection.gestures[i][0]
                h_data.gesture_name = top_gesture.category_name
                h_data.gesture_score = round(top_gesture.score, 3)

            # Convert 21 landmarks
            pts = [(lm.x, lm.y, lm.z) for lm in landmarks]
            h_data.landmarks = pts

            h_data.wrist = (landmarks[0].x, landmarks[0].y)
            h_data.thumb_tip = (landmarks[4].x, landmarks[4].y)
            h_data.index_tip = (landmarks[8].x, landmarks[8].y)
            h_data.middle_tip = (landmarks[12].x, landmarks[12].y)

            # 3D Depth & relative metrics for forward tap and flick detection
            h_data.index_tip_z = landmarks[8].z
            h_data.middle_tip_z = landmarks[12].z
            # Forward tap depth: index tip relative to index MCP knuckle (5)
            h_data.index_rel_z = landmarks[8].z - landmarks[5].z
            # Middle finger extension distance: distance between middle tip (12) and middle MCP (9)
            m_dx = landmarks[12].x - landmarks[9].x
            m_dy = landmarks[12].y - landmarks[9].y
            h_data.middle_dist = math.hypot(m_dx, m_dy)

            # Fingertip bend metrics for instantaneous click (5=MCP, 6=PIP, 7=DIP, 8=TIP)
            if len(landmarks) >= 9:
                p5 = landmarks[5]
                p6 = landmarks[6]
                p7 = landmarks[7]
                p8 = landmarks[8]
                d8_6 = math.hypot(p8.x - p6.x, p8.y - p6.y, p8.z - p6.z)
                d8_7 = math.hypot(p8.x - p7.x, p8.y - p7.y, p8.z - p7.z)
                d7_6 = math.hypot(p7.x - p6.x, p7.y - p6.y, p7.z - p6.z)
                d6_5 = math.hypot(p6.x - p5.x, p6.y - p5.y, p6.z - p5.z)
                d8_5 = math.hypot(p8.x - p5.x, p8.y - p5.y, p8.z - p5.z)

                seg_pip_dip_tip = d8_7 + d7_6
                total_index_len = seg_pip_dip_tip + d6_5

                if seg_pip_dip_tip > 1e-4:
                    h_data.index_bend_ratio = round(d8_6 / seg_pip_dip_tip, 3)
                if total_index_len > 1e-4:
                    h_data.index_curl_ratio = round(d8_5 / total_index_len, 3)

            # Palm Center: average of wrist (0), index MCP (5), middle MCP (9), pinky MCP (17)
            h_data.center_x = (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[17].x) / 4.0
            h_data.center_y = (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[17].y) / 4.0

            # Scale reference: Distance between wrist and middle MCP
            palm_scale = math.hypot(
                (landmarks[9].x - landmarks[0].x) * w,
                (landmarks[9].y - landmarks[0].y) * h
            )
            if palm_scale < 1e-4:
                palm_scale = 1.0

            # Pinch distance: Distance between thumb tip (4) and index tip (8)
            thumb_x, thumb_y = landmarks[4].x * w, landmarks[4].y * h
            index_x, index_y = landmarks[8].x * w, landmarks[8].y * h
            raw_pinch_dist = math.hypot(thumb_x - index_x, thumb_y - index_y)
            norm_pinch = raw_pinch_dist / palm_scale

            h_data.pinch_distance = round(norm_pinch, 3)
            h_data.is_pinched = norm_pinch < 0.28

            res.hands.append(h_data)

        # Primary hand is the first detected or most prominent hand
        if res.hands:
            res.primary_hand = res.hands[0]

        if draw_overlay:
            self._draw_hud(annotated_frame, res, w, h, extra_info)

        return res, annotated_frame

    def _draw_hud(self, img: np.ndarray, res: HandDetectionResult, w: int, h: int, extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Draws glowing cyberpunk hand skeleton, air mouse reticles, and indicators.
        """
        CYAN = (240, 200, 0)
        MAGENTA = (200, 50, 255)
        GREEN = (50, 255, 120)
        AMBER = (0, 165, 255)
        WHITE = (255, 255, 255)

        for hand in res.hands:
            pts_px = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand.landmarks]

            # Draw bones / connections
            for p1_idx, p2_idx in self.HAND_CONNECTIONS:
                pt1 = pts_px[p1_idx]
                pt2 = pts_px[p2_idx]
                cv2.line(img, pt1, pt2, (120, 100, 30), 2, cv2.LINE_AA)
                cv2.line(img, pt1, pt2, CYAN, 1, cv2.LINE_AA)

            # Draw fingertips
            for idx, pt in enumerate(pts_px):
                if idx in (4, 8, 12, 16, 20):
                    cv2.circle(img, pt, 6, GREEN, -1, cv2.LINE_AA)
                    cv2.circle(img, pt, 8, WHITE, 1, cv2.LINE_AA)
                else:
                    cv2.circle(img, pt, 3, CYAN, -1, cv2.LINE_AA)

            # Palm Center & Arc-Reactor Reticle
            cx, cy = int(hand.center_x * w), int(hand.center_y * h)
            cv2.drawMarker(img, (cx, cy), GREEN, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
            # Concentric ring reticle
            cv2.circle(img, (cx, cy), 18, CYAN, 1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 24, AMBER, 1, cv2.LINE_AA)
            # 4 Tick marks
            cv2.line(img, (cx - 28, cy), (cx - 20, cy), AMBER, 1, cv2.LINE_AA)
            cv2.line(img, (cx + 20, cy), (cx + 28, cy), AMBER, 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy - 28), (cx, cy - 20), AMBER, 1, cv2.LINE_AA)
            cv2.line(img, (cx, cy + 20), (cx, cy + 28), AMBER, 1, cv2.LINE_AA)

        # Dual-Hand Hologram Energy Link
        if len(res.hands) >= 2:
            h1, h2 = res.hands[0], res.hands[1]
            c1x, c1y = int(h1.center_x * w), int(h1.center_y * h)
            c2x, c2y = int(h2.center_x * w), int(h2.center_y * h)
            dist_px = int(math.hypot(c2x - c1x, c2y - c1y))

            # Double-layer holographic laser beam
            cv2.line(img, (c1x, c1y), (c2x, c2y), AMBER, 3, cv2.LINE_AA)
            cv2.line(img, (c1x, c1y), (c2x, c2y), CYAN, 2, cv2.LINE_AA)
            cv2.line(img, (c1x, c1y), (c2x, c2y), WHITE, 1, cv2.LINE_AA)

            # Midpoint Holographic HUD Badge
            mx, my = (c1x + c2x) // 2, (c1y + c2y) // 2
            badge_text = f"DUAL-HAND LINK: {dist_px}px"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (mx - tw // 2 - 8, my - th - 8), (mx + tw // 2 + 8, my + 8), (15, 17, 24), -1)
            cv2.rectangle(img, (mx - tw // 2 - 8, my - th - 8), (mx + tw // 2 + 8, my + 8), CYAN, 1)
            cv2.putText(img, badge_text, (mx - tw // 2, my), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1, cv2.LINE_AA)

            sub_text = "SPREAD: EXPAND | PINCH: SHRINK"
            (stw, sth), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            cv2.putText(img, sub_text, (mx - stw // 2, my + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, AMBER, 1, cv2.LINE_AA)

        # Air Mouse Overlay
        if res.primary_hand and extra_info:
            ph = res.primary_hand
            ix, iy = int(ph.index_tip[0] * w), int(ph.index_tip[1] * h)
            air_mouse_active = extra_info.get("air_mouse_active", False)
            idle_progress = extra_info.get("idle_progress", 0.0)
            dwell_progress = extra_info.get("dwell_progress", 0.0)
            cursor_pos = extra_info.get("cursor_pos", (0, 0))

            if air_mouse_active:
                # Glowing Cyber Air Mouse Reticle
                cv2.circle(img, (ix, iy), 16, CYAN, 2, cv2.LINE_AA)
                cv2.circle(img, (ix, iy), 4, GREEN, -1, cv2.LINE_AA)
                cv2.line(img, (ix - 12, iy), (ix - 5, iy), CYAN, 1, cv2.LINE_AA)
                cv2.line(img, (ix + 5, iy), (ix + 12, iy), CYAN, 1, cv2.LINE_AA)
                cv2.line(img, (ix, iy - 12), (ix, iy - 5), CYAN, 1, cv2.LINE_AA)
                cv2.line(img, (ix, iy + 5), (ix, iy + 12), CYAN, 1, cv2.LINE_AA)

                # Dwell Click Radial Progress Arc
                if dwell_progress > 0.03:
                    deg = int(360.0 * min(1.0, dwell_progress))
                    cv2.ellipse(img, (ix, iy), (22, 22), 0, -90, -90 + deg, AMBER, 3, cv2.LINE_AA)

                click_feedback = extra_info.get("click_feedback", None)
                if click_feedback:
                    ring_col = GREEN if click_feedback == "left" else MAGENTA
                    cv2.circle(img, (ix, iy), 25, ring_col, 2, cv2.LINE_AA)
                    cv2.circle(img, (ix, iy), 32, WHITE, 1, cv2.LINE_AA)

                cv2.putText(img, f"AIR MOUSE [{cursor_pos[0]},{cursor_pos[1]}]", (ix + 18, iy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, CYAN, 1, cv2.LINE_AA)

            elif idle_progress > 0.08:
                # Idle Activation Charging Ring
                deg = int(360.0 * min(1.0, idle_progress))
                cv2.ellipse(img, (ix, iy), (18, 18), 0, -90, -90 + deg, MAGENTA, 2, cv2.LINE_AA)
                cv2.putText(img, f"AIR MOUSE {int(idle_progress * 100)}%", (ix + 18, iy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, MAGENTA, 1, cv2.LINE_AA)

        # Bottom Mini HUD
        if res.primary_hand:
            ph = res.primary_hand
            cv2.rectangle(img, (10, h - 38), (w - 10, h - 10), (20, 20, 25), -1)
            cv2.rectangle(img, (10, h - 38), (w - 10, h - 10), (60, 60, 80), 1)

            state_desc = extra_info.get("engine_state", "") if extra_info else ""
            pose_name = ph.gesture_name if ph.gesture_name != "None" else "TRACKING"
            if state_desc:
                text = f"[{state_desc}] | POSE: {pose_name} | LATENCY: {res.latency_ms}ms"
            else:
                text = f"POSE: {pose_name} | HANDS: {res.num_hands} | LATENCY: {res.latency_ms}ms"
            cv2.putText(img, text, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    def close(self):
        if hasattr(self, "recognizer") and self.recognizer:
            self.recognizer.close()
