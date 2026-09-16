"""
Automated Test Suite for BlinkFlow
"""

import json
import os
import tempfile
import time
import unittest

from blink_core import (
    ActionConfig,
    ActionDispatcher,
    AppConfig,
    BlinkDetector,
    ConfigManager,
    DetectionResult,
    FaceGesture,
    GestureEngine,
    SensitivityConfig,
)
from fastapi.testclient import TestClient
from server.app import app


class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.tmp_dir, "test_config.json")
        self.cm = ConfigManager(self.config_file)

    def test_default_config_creation(self):
        cfg = self.cm.load()
        self.assertTrue(cfg.active)
        self.assertIn("DOUBLE_BLINK", cfg.mappings)
        self.assertEqual(cfg.mappings["DOUBLE_BLINK"].value, "Page_Down")

    def test_update_config(self):
        self.cm.update_from_dict({
            "active": False,
            "sensitivity": {"blink_threshold": 0.65}
        })
        reloaded = self.cm.load()
        self.assertFalse(reloaded.active)
        self.assertAlmostEqual(reloaded.sensitivity.blink_threshold, 0.65)

    def test_apply_preset(self):
        res = self.cm.apply_preset("browser")
        self.assertTrue(res)
        reloaded = self.cm.load()
        self.assertEqual(reloaded.mappings["DOUBLE_BLINK"].value, "ctrl+t")


class TestGestureEngine(unittest.TestCase):
    def setUp(self):
        self.sens = SensitivityConfig(
            blink_threshold=0.55,
            wink_open_threshold=0.35,
            min_blink_duration_ms=50,
            long_blink_duration_ms=300,
            double_blink_window_ms=350,
            cooldown_ms=200
        )
        self.engine = GestureEngine(self.sens)
        self.detected_gestures = []
        self.engine.on_gesture_detected = lambda g, d: self.detected_gestures.append(g)

    def test_no_face_resets_state(self):
        self.engine.process(0.9, 0.9, False)
        self.assertEqual(self.engine.current_state_text, "NO_FACE")

    def test_double_blink_detection(self):
        # 1st Blink: close for 80ms, then open
        self.engine.process(0.85, 0.85, True)
        time.sleep(0.08)
        self.engine.process(0.1, 0.1, True)

        # Gap between blinks
        time.sleep(0.08)

        # 2nd Blink: close for 80ms, then open
        self.engine.process(0.85, 0.85, True)
        time.sleep(0.08)
        self.engine.process(0.1, 0.1, True)

        # Wait for double blink window to expire and settle
        time.sleep(0.38)
        self.engine.process(0.1, 0.1, True)

        self.assertIn(FaceGesture.DOUBLE_BLINK, self.detected_gestures)

    def test_long_blink_hold(self):
        self.detected_gestures.clear()
        # Close eyes
        self.engine.process(0.85, 0.85, True)
        # Hold longer than long_blink_duration_ms (300ms)
        time.sleep(0.35)
        res = self.engine.process(0.85, 0.85, True)
        self.assertEqual(res, FaceGesture.LONG_BLINK)
        self.assertIn(FaceGesture.LONG_BLINK, self.detected_gestures)

    def test_left_wink_detection(self):
        self.detected_gestures.clear()
        # Left closed, right open
        self.engine.process(0.85, 0.1, True)
        time.sleep(0.12)
        # Release to open
        self.engine.process(0.1, 0.1, True)
        self.assertIn(FaceGesture.LEFT_WINK, self.detected_gestures)

    def test_right_wink_detection(self):
        self.detected_gestures.clear()
        # Right closed, left open
        self.engine.process(0.1, 0.85, True)
        time.sleep(0.12)
        # Release to open
        self.engine.process(0.1, 0.1, True)
        self.assertIn(FaceGesture.RIGHT_WINK, self.detected_gestures)


class TestActionDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = ActionDispatcher()

    def test_command_execution(self):
        # Run a harmless echo test
        success = self.dispatcher._execute_sync(
            gesture="TEST",
            action_type="command",
            action_value="echo 'BlinkFlow Test'"
        )
        self.assertTrue(success)
        history = self.dispatcher.get_recent_history()
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0]["gesture"], "TEST")

    def test_xdotool_formatting(self):
        fmt = self.dispatcher._format_for_xdotool("ctrl + Shift + Tab")
        self.assertEqual(fmt, "ctrl+shift+Tab")
        fmt2 = self.dispatcher._format_for_xdotool("page_down")
        self.assertEqual(fmt2, "Page_Down")


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_status(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("active", data)
        self.assertIn("fps", data)

    def test_get_and_post_config(self):
        res = self.client.get("/api/config")
        self.assertEqual(res.status_code, 200)

        update_res = self.client.post("/api/config", json={"active": True})
        self.assertEqual(update_res.status_code, 200)

    def test_test_action_api(self):
        res = self.client.post("/api/test-action", json={
            "gesture": "DOUBLE_BLINK",
            "type": "command",
            "value": "echo 'Testing API Dispatch'",
            "description": "API Test"
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["dispatched"])

    def test_preset_api(self):
        res = self.client.post("/api/preset", json={"preset": "media"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["preset"], "media")


if __name__ == "__main__":
    unittest.main()
