"""
Thread-safe OpenCV Camera capture worker with auto-reconnect and FPS metering.
"""

import glob
import logging
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("CameraWorker")


class CameraWorker:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._last_fps_time = time.time()
        self.actual_fps = 0.0
        self.connected = False

    @staticmethod
    def list_available_cameras() -> List[int]:
        """
        Detects video capture devices on Linux.
        """
        available = []
        video_devices = sorted(glob.glob("/dev/video*"))
        for dev in video_devices:
            try:
                # Extract index from /dev/videoN
                idx = int(dev.replace("/dev/video", ""))
                # Quick test
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(idx)
                    cap.release()
            except Exception:
                pass
        return available if available else [0]

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            self._running = True

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        with self._lock:
            self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
            self.connected = False

    def change_camera(self, new_index: int) -> bool:
        if new_index == self.camera_index and self.connected:
            return True
        logger.info(f"Switching camera from {self.camera_index} to {new_index}")
        self.stop()
        self.camera_index = new_index
        return self.start()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _capture_loop(self) -> None:
        logger.info(f"Opening camera index {self.camera_index}...")
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not self._cap.isOpened():
            logger.error(f"Failed to open camera index {self.camera_index}")
            self.connected = False
            return

        self.connected = True
        logger.info(f"Camera {self.camera_index} successfully opened.")

        fps_counter = 0
        fps_start = time.time()

        while True:
            with self._lock:
                if not self._running:
                    break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Mirror frame horizontally for intuitive mirror-like webcam view
            frame = cv2.flip(frame, 1)

            with self._lock:
                self._latest_frame = frame
                self._frame_count += 1

            fps_counter += 1
            now = time.time()
            if now - fps_start >= 1.0:
                self.actual_fps = round(fps_counter / (now - fps_start), 1)
                fps_counter = 0
                fps_start = now

            # Sleep slightly to match target FPS
            time.sleep(1.0 / (self.target_fps * 1.5))

        if self._cap:
            self._cap.release()
            self._cap = None
        self.connected = False
        logger.info("Camera capture loop terminated.")
