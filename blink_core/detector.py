"""
Face and Eye Blink Detector using MediaPipe FaceLandmarker with Blendshapes and EAR metrics.
"""

import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


@dataclass
class DetectionResult:
    face_detected: bool = False
    # Blink scores from 0.0 (wide open) to 1.0 (fully closed)
    left_blink_score: float = 0.0
    right_blink_score: float = 0.0
    # Eye Aspect Ratio (higher = more open, lower = closed)
    left_ear: float = 0.0
    right_ear: float = 0.0
    # Iris / Eye center coordinates (normalized x, y)
    left_eye_center: Tuple[float, float] = (0.0, 0.0)
    right_eye_center: Tuple[float, float] = (0.0, 0.0)
    # Mouth / Lips center (normalized x, y) for Shh / Quiet gesture
    mouth_center: Tuple[float, float] = (0.0, 0.0)
    # Face Bounding Box in pixel coordinates (x1, y1, x2, y2)
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    # Processing latency
    latency_ms: float = 0.0
    # Eye contours for client-side rendering (normalized [x, y])
    left_eye_contour: List[Tuple[float, float]] = field(default_factory=list)
    right_eye_contour: List[Tuple[float, float]] = field(default_factory=list)


class BlinkDetector:
    # Key eye contour landmarks in MediaPipe 468-point mesh
    # Anatomical Left Eye (subject's left)
    LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_EAR_POINTS = [362, 385, 387, 263, 373, 380] # [p1, p2, p3, p4, p5, p6]
    
    # Anatomical Right Eye (subject's right)
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EAR_POINTS = [33, 160, 158, 133, 153, 144]  # [p1, p2, p3, p4, p5, p6]

    def __init__(self, model_path: str = "face_landmarker.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe Face Landmarker model not found at {model_path}")

        self.model_path = model_path
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _calculate_ear(self, landmarks, points: List[int], w: int, h: int) -> float:
        """
        Calculates Eye Aspect Ratio:
        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
        """
        try:
            coords = []
            for idx in points:
                lm = landmarks[idx]
                coords.append((lm.x * w, lm.y * h))

            p1, p2, p3, p4, p5, p6 = coords
            # Vertical distances
            d_v1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
            d_v2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
            # Horizontal distance
            d_h = math.hypot(p1[0] - p4[0], p1[1] - p4[1])

            if d_h == 0:
                return 0.0
            return (d_v1 + d_v2) / (2.0 * d_h)
        except Exception:
            return 0.0

    def process_frame(self, frame_bgr: np.ndarray, draw_overlay: bool = True) -> Tuple[DetectionResult, np.ndarray]:
        t0 = time.time()
        h, w, _ = frame_bgr.shape
        res = DetectionResult()

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection = self.landmarker.detect(mp_image)
        latency = (time.time() - t0) * 1000.0
        res.latency_ms = round(latency, 1)

        annotated_frame = frame_bgr.copy() if draw_overlay else frame_bgr

        if not detection.face_landmarks or len(detection.face_landmarks) == 0:
            res.face_detected = False
            return res, annotated_frame

        res.face_detected = True
        landmarks = detection.face_landmarks[0]

        # 1. Extract Blendshape scores
        left_blend = 0.0
        right_blend = 0.0
        if detection.face_blendshapes and len(detection.face_blendshapes) > 0:
            for cat in detection.face_blendshapes[0]:
                if cat.category_name == "eyeBlinkLeft":
                    left_blend = float(cat.score)
                elif cat.category_name == "eyeBlinkRight":
                    right_blend = float(cat.score)

        res.left_blink_score = round(left_blend, 3)
        res.right_blink_score = round(right_blend, 3)

        # 2. Extract EAR
        res.left_ear = round(self._calculate_ear(landmarks, self.LEFT_EAR_POINTS, w, h), 3)
        res.right_ear = round(self._calculate_ear(landmarks, self.RIGHT_EAR_POINTS, w, h), 3)

        # 3. Extract Eye Contours & Centers
        left_pts = [(landmarks[idx].x, landmarks[idx].y) for idx in self.LEFT_EYE_INDICES]
        right_pts = [(landmarks[idx].x, landmarks[idx].y) for idx in self.RIGHT_EYE_INDICES]
        res.left_eye_contour = left_pts
        res.right_eye_contour = right_pts

        # Centers
        res.left_eye_center = (
            sum(p[0] for p in left_pts) / len(left_pts),
            sum(p[1] for p in left_pts) / len(left_pts)
        )
        res.right_eye_center = (
            sum(p[0] for p in right_pts) / len(right_pts),
            sum(p[1] for p in right_pts) / len(right_pts)
        )

        # Mouth / Lips Center (Upper Lip 13, Lower Lip 14)
        if len(landmarks) > 14:
            res.mouth_center = (
                (landmarks[13].x + landmarks[14].x) / 2.0,
                (landmarks[13].y + landmarks[14].y) / 2.0
            )

        # 4. Compute Face Bounding Box with margin for anonymity blurring
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        fw = x_max - x_min
        fh = y_max - y_min
        x1 = max(0, int(x_min - 0.12 * fw))
        y1 = max(0, int(y_min - 0.18 * fh))
        x2 = min(w, int(x_max + 0.12 * fw))
        y2 = min(h, int(y_max + 0.10 * fh))
        res.face_bbox = (x1, y1, x2, y2)

        # 5. Draw Overlay if enabled
        if draw_overlay:
            self._draw_hud(annotated_frame, res, landmarks, w, h)

        return res, annotated_frame

    @staticmethod
    def apply_face_blur(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Applies a clean privacy shield (Gaussian blur + mosaic) over the face bounding box.
        """
        x1, y1, x2, y2 = bbox
        h_img, w_img, _ = img.shape
        x1 = max(0, min(w_img - 1, x1))
        x2 = max(0, min(w_img, x2))
        y1 = max(0, min(h_img - 1, y1))
        y2 = max(0, min(h_img, y2))
        rw = x2 - x1
        rh = y2 - y1
        if rw < 4 or rh < 4:
            return img

        roi = img[y1:y2, x1:x2]
        # Pixelate then Gaussian blur for 100% anonymization
        mw, mh = max(8, rw // 14), max(8, rh // 14)
        small = cv2.resize(roi, (mw, mh), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        k = max(15, (min(rw, rh) // 6) * 2 + 1)
        blurred = cv2.GaussianBlur(mosaic, (k, k), 25)

        # Elliptical mask for smooth organic blending around the head
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 255, -1)
        mask_3d = (mask[:, :, None] == 255)

        img[y1:y2, x1:x2] = np.where(mask_3d, blurred, roi)
        return img

    def _draw_hud(self, img: np.ndarray, res: DetectionResult, landmarks, w: int, h: int) -> None:
        """
        Draws sleek cyber/HUD indicators on the frame.
        """
        # Colors (BGR)
        CYAN = (240, 200, 0)
        MAGENTA = (200, 50, 255)
        GREEN = (50, 255, 120)
        AMBER = (0, 165, 255)
        WHITE = (255, 255, 255)

        # Draw eye contours
        def draw_contour(indices, blink_score):
            pts = []
            for idx in indices:
                lm = landmarks[idx]
                pts.append((int(lm.x * w), int(lm.y * h)))
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            # Closed eye glows magenta/amber, open eye is neon cyan
            color = MAGENTA if blink_score > 0.5 else CYAN
            thickness = 2 if blink_score > 0.5 else 1
            cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        draw_contour(self.LEFT_EYE_INDICES, res.left_blink_score)
        draw_contour(self.RIGHT_EYE_INDICES, res.right_blink_score)

        # Draw Iris / Pupil Center
        lx, ly = int(res.left_eye_center[0] * w), int(res.left_eye_center[1] * h)
        rx, ry = int(res.right_eye_center[0] * w), int(res.right_eye_center[1] * h)
        cv2.circle(img, (lx, ly), 3, GREEN if res.left_blink_score < 0.5 else MAGENTA, -1)
        cv2.circle(img, (rx, ry), 3, GREEN if res.right_blink_score < 0.5 else MAGENTA, -1)

        # Bottom mini HUD bar
        cv2.rectangle(img, (10, h - 38), (w - 10, h - 10), (20, 20, 25), -1)
        cv2.rectangle(img, (10, h - 38), (w - 10, h - 10), (60, 60, 80), 1)

        l_pct = int(res.left_blink_score * 100)
        r_pct = int(res.right_blink_score * 100)
        hud_text = f"L: {l_pct}% | R: {r_pct}% | Latency: {res.latency_ms}ms"
        cv2.putText(img, hud_text, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    def close(self):
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()
