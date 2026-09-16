"""
Automated Test Suite for HandFlow Gestures and Continuous Actions
"""

import time
import unittest
from unittest import mock

import numpy as np

from blink_core import (
    ActionConfig,
    ActionDispatcher,
    AppConfig,
    BlinkDetector,
    ConfigManager,
    HandAction,
    HandData,
    HandDetectionResult,
    HandDetector,
    HandGestureEngine,
    HandSensitivityConfig,
)
from fastapi.testclient import TestClient
from server.app import app


class TestHandGestureEngine(unittest.TestCase):
    def setUp(self):
        self.engine = HandGestureEngine(
            scroll_deadzone=0.025,
            scroll_inverted=False,
            input_delay_ms=100,
            swipe_velocity_thresh=0.40,
            swipe_min_dist=0.10,
            pose_hold_ms=100
        )
        self.continuous_actions = []
        self.discrete_actions = []

        self.engine.on_continuous_action = lambda act, mag: self.continuous_actions.append((act, mag))
        self.engine.on_discrete_gesture = lambda act, meta: self.discrete_actions.append((act, meta))

        # Seed initial hand presence
        t_init = time.time() - 0.30
        self.engine.hand_entered_time = t_init

    def test_scroll_up_down(self):
        # Initial position at y=0.5
        h0 = HandData(center_x=0.5, center_y=0.5, pinch_distance=0.5)
        self.engine.process(h0)
        time.sleep(0.04)

        # Move hand UP across several frames
        act = None
        for y in [0.46, 0.42, 0.38]:
            res = self.engine.process(HandData(center_x=0.5, center_y=y, pinch_distance=0.5))
            if res: act = res
            time.sleep(0.035)

        self.assertEqual(act, HandAction.SCROLL_UP)
        self.assertTrue(any(a[0] == HandAction.SCROLL_UP for a in self.continuous_actions))

        # Wait past input delay cooldown (200ms)
        time.sleep(0.22)

        # Move hand DOWN across several frames
        act2 = None
        for y in [0.44, 0.50, 0.56]:
            res = self.engine.process(HandData(center_x=0.5, center_y=y, pinch_distance=0.5))
            if res: act2 = res
            time.sleep(0.035)

        self.assertEqual(act2, HandAction.SCROLL_DOWN)
        self.assertTrue(any(a[0] == HandAction.SCROLL_DOWN for a in self.continuous_actions))

    def test_no_pinch_zoom(self):
        """Ensure pinch movements do not trigger zoom actions or pinch clicks."""
        self.continuous_actions.clear()
        self.discrete_actions.clear()

        # Seed initial hand with pinch distance 0.50
        self.engine.process(HandData(center_x=0.5, center_y=0.5, pinch_distance=0.50))
        time.sleep(0.04)

        # Simulate pinch in, pinch out, pinch click
        for p in [0.40, 0.30, 0.20, 0.10, 0.05, 0.50, 0.80]:
            res = self.engine.process(HandData(center_x=0.5, center_y=0.5, pinch_distance=p, is_pinched=p < 0.2))
            # Res should never be zoom or pinch click
            self.assertNotIn(res, [HandAction.ZOOM_IN, HandAction.ZOOM_OUT, HandAction.PINCH_CLICK])
            time.sleep(0.02)

        self.assertFalse(any(a[0] in [HandAction.ZOOM_IN, HandAction.ZOOM_OUT, HandAction.PINCH_CLICK] for a in self.continuous_actions))
        self.assertFalse(any(a[0] in [HandAction.ZOOM_IN, HandAction.ZOOM_OUT, HandAction.PINCH_CLICK] for a in self.discrete_actions))

    def test_input_delay_cooldown(self):
        """Verify inputs within the cooldown window are suppressed with DELAY status."""
        self.continuous_actions.clear()
        self.engine.input_delay_ms = 250  # 250ms cooldown

        # Trigger first action (Scroll UP)
        h0 = HandData(center_x=0.5, center_y=0.5, pinch_distance=0.5)
        self.engine.process(h0)
        time.sleep(0.03)

        act1 = None
        for y in [0.46, 0.40, 0.35]:
            r = self.engine.process(HandData(center_x=0.5, center_y=y, pinch_distance=0.5))
            if r: act1 = r
            time.sleep(0.02)

        self.assertEqual(act1, HandAction.SCROLL_UP)

        # Immediately try to trigger another action at 50ms into 250ms cooldown
        time.sleep(0.05)
        blocked = self.engine.process(HandData(center_x=0.5, center_y=0.55, pinch_distance=0.5))
        self.assertIsNone(blocked, "Action during cooldown must be None")
        self.assertTrue("DELAY" in self.engine.current_state_text, f"Expected DELAY in state text, got {self.engine.current_state_text}")

        # Wait out remainder of cooldown (> 250ms total)
        time.sleep(0.22)

        # Now trigger next action (Scroll DOWN)
        act2 = None
        for y in [0.58, 0.64, 0.70]:
            r = self.engine.process(HandData(center_x=0.5, center_y=y, pinch_distance=0.5))
            if r: act2 = r
            time.sleep(0.02)

        self.assertEqual(act2, HandAction.SCROLL_DOWN)

    def test_swipe_horizontal(self):
        self.discrete_actions.clear()
        # 1. Fast move to right: Swipe Right
        self.engine.process(HandData(center_x=0.2, center_y=0.5, pinch_distance=0.5))
        time.sleep(0.04)
        self.engine.process(HandData(center_x=0.30, center_y=0.5, pinch_distance=0.5))
        time.sleep(0.04)
        act_r = self.engine.process(HandData(center_x=0.58, center_y=0.5, pinch_distance=0.5))
        self.assertEqual(act_r, HandAction.SWIPE_RIGHT)
        self.assertTrue(any(a[0] == HandAction.SWIPE_RIGHT for a in self.discrete_actions))

        # Wait past swipe cooldown
        time.sleep(0.40)
        self.discrete_actions.clear()

        # 2. Fast move to left: Swipe Left
        self.engine.process(HandData(center_x=0.8, center_y=0.5, pinch_distance=0.5))
        time.sleep(0.04)
        self.engine.process(HandData(center_x=0.70, center_y=0.5, pinch_distance=0.5))
        time.sleep(0.04)
        act_l = self.engine.process(HandData(center_x=0.42, center_y=0.5, pinch_distance=0.5))
        self.assertEqual(act_l, HandAction.SWIPE_LEFT)
        self.assertTrue(any(a[0] == HandAction.SWIPE_LEFT for a in self.discrete_actions))

    def test_static_pose_hold(self):
        self.discrete_actions.clear()
        # Hold Open_Palm with high confidence
        h0 = HandData(center_x=0.5, center_y=0.5, gesture_name="Open_Palm", gesture_score=0.9, pinch_distance=0.8)
        self.engine.process(h0)

        # Still holding after hold duration
        time.sleep(0.12)
        h1 = HandData(center_x=0.5, center_y=0.5, gesture_name="Open_Palm", gesture_score=0.9, pinch_distance=0.8)
        act = self.engine.process(h1)
        self.assertEqual(act, HandAction.OPEN_PALM)
        self.assertTrue(any(a[0] == HandAction.OPEN_PALM for a in self.discrete_actions))

    def test_air_mouse_idle_activation(self):
        """Verify that holding index finger steady activates Air Mouse mode."""
        cursor_positions = []
        self.engine.air_mouse_enabled = True
        self.engine.air_mouse_idle_sec = 0.20  # Fast 200ms for testing
        self.engine.on_cursor_move = lambda sx, sy: cursor_positions.append((sx, sy))

        # Initial steady finger
        h0 = HandData(center_x=0.5, center_y=0.5, index_tip=(0.5, 0.5))
        self.engine.process(h0)
        self.assertFalse(self.engine.air_mouse_active)

        # Wait past idle activation duration
        time.sleep(0.22)
        h1 = HandData(center_x=0.5, center_y=0.5, index_tip=(0.5, 0.5))
        self.engine.process(h1)

        self.assertTrue(self.engine.air_mouse_active)
        self.assertEqual(self.engine.idle_progress, 1.0)
        self.assertIn("AIR MOUSE", self.engine.current_state_text)

    def test_air_mouse_cursor_movement(self):
        """Verify that moving fingertip drives screen cursor coordinates smoothly."""
        cursor_positions = []
        self.engine.air_mouse_enabled = True
        self.engine.air_mouse_active = True
        self.engine.set_screen_resolution(1920, 1080)
        self.engine.on_cursor_move = lambda sx, sy: cursor_positions.append((sx, sy))

        # Move finger to right/center
        h = HandData(center_x=0.6, center_y=0.4, index_tip=(0.60, 0.40))
        self.engine.process(h)

        self.assertTrue(len(cursor_positions) > 0)
        last_x, last_y = cursor_positions[-1]
        self.assertTrue(0 <= last_x <= 1920)
        self.assertTrue(0 <= last_y <= 1080)

    def test_air_mouse_dwell_click(self):
        """Verify dwell click only fires when dwell_click_enabled is explicitly True."""
        clicks = []
        self.engine.air_mouse_enabled = True
        self.engine.air_mouse_active = True
        self.engine.dwell_click_enabled = False  # Disabled by default per user specification
        self.engine.dwell_click_ms = 100
        self.engine.on_cursor_click = lambda btn: clicks.append(btn)

        h0 = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50))
        self.engine.process(h0)
        time.sleep(0.12)
        h1 = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50))
        self.engine.process(h1)
        self.assertEqual(len(clicks), 0, "Dwell click must NOT trigger when dwell_click_enabled is False")

        # Now enable dwell click explicitly
        self.engine.dwell_click_enabled = True
        self.engine.dwell_clicked = False
        self.engine.dwell_start_time = None
        self.engine.process(h0)
        time.sleep(0.12)
        self.engine.process(h1)
        self.assertIn("left", clicks)
        self.assertTrue(self.engine.dwell_clicked)

    def test_air_mouse_fingertip_bend_instant_click(self):
        """Verify that bending the index fingertip triggers an instantaneous left click with zero delay."""
        clicks = []
        self.engine.air_mouse_enabled = True
        self.engine.air_mouse_active = True
        self.engine.on_cursor_click = lambda btn: clicks.append(btn)

        # Frame 1: Index finger straight (bend ratio ~0.98)
        h_straight = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50), index_bend_ratio=0.98)
        self.engine.process(h_straight)
        self.assertEqual(len(clicks), 0)
        self.assertTrue(self.engine.finger_click_ready)

        # Frame 2: Index fingertip bent down (bend ratio 0.72 < 0.78 threshold)
        h_bent = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50), index_bend_ratio=0.72)
        self.engine.process(h_bent)

        # Must fire immediate left click without waiting for any dwell timer!
        self.assertIn("left", clicks)
        self.assertEqual(len(clicks), 1)
        self.assertIn("FINGER BEND CLICK", self.engine.current_state_text)
        self.assertFalse(self.engine.finger_click_ready)
        self.assertGreater(self.engine.cursor_freeze_until, time.time())

        # Frame 3: Finger still held bent -> debounce must prevent spamming
        self.engine.process(h_bent)
        self.assertEqual(len(clicks), 1, "Click must be debounced while finger remains bent")

        # Frame 4: Finger unbends / straightens back up (> 0.84) -> re-arms click
        self.engine.process(h_straight)
        self.assertTrue(self.engine.finger_click_ready, "finger_click_ready must reset when unbent")

    def test_air_mouse_peace_right_click(self):
        """Verify that displaying Peace sign (Victory) triggers right-click."""
        clicks = []
        self.engine.air_mouse_enabled = True
        self.engine.air_mouse_active = True
        self.engine.on_cursor_click = lambda btn: clicks.append(btn)

        h_peace = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50), gesture_name="Victory", gesture_score=0.95)
        self.engine.process(h_peace)

        self.assertIn("right", clicks)

    def test_air_mouse_forward_tap_click(self):
        """Verify that poking/tapping the index finger forward triggers a left click."""
        clicks = []
        self.engine.air_mouse_enabled = True
        self.engine.air_mouse_active = True
        self.engine.on_cursor_click = lambda btn: clicks.append(btn)

        # Baseline frame: finger at rest (index_rel_z = 0.0)
        h0 = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50), index_rel_z=0.0)
        self.engine.process(h0)

        time.sleep(0.06)

        # Forward tap frame: finger poked forward rapidly (index_rel_z = -0.05, velocity ~ -0.8)
        h1 = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50), index_rel_z=-0.05)
        self.engine.process(h1)

        self.assertIn("left", clicks)
        self.assertIn("FORWARD TAP CLICK", self.engine.current_state_text)
        self.assertGreater(self.engine.cursor_freeze_until, time.time())

    def test_air_mouse_middle_finger_flick_right_click(self):
        """Verify that flicking the middle finger outward triggers a right click."""
        clicks = []
        self.engine.air_mouse_enabled = True
        self.engine.air_mouse_active = True
        self.engine.on_cursor_click = lambda btn: clicks.append(btn)

        # Baseline frame: middle finger curled/retracted (middle_dist = 0.08)
        h0 = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50), middle_dist=0.08)
        self.engine.process(h0)

        time.sleep(0.06)

        # Flick frame: middle finger extended rapidly (middle_dist = 0.18, velocity ~ 1.6)
        h1 = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50), middle_dist=0.18)
        self.engine.process(h1)

        self.assertIn("right", clicks)
        self.assertIn("MIDDLE FLICK RIGHT CLICK", self.engine.current_state_text)
        self.assertGreater(self.engine.cursor_freeze_until, time.time())

    def test_air_mouse_precision_deadband(self):
        """Verify that micro-jitter movements under deadband threshold (<3.0px) do not drift cursor."""
        self.engine.air_mouse_enabled = True
        self.engine.air_mouse_active = True

        # Initial position
        h0 = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50))
        self.engine.process(h0)
        initial_x = self.engine.smooth_cursor_x
        initial_y = self.engine.smooth_cursor_y
        self.assertIsNotNone(initial_x)

        # Micro-tremor / jitter: tiny 0.0005 displacement in normalized coords
        # (0.0005 / 0.70 * 1920 = ~1.37 pixels displacement, under deadband_px=3.0)
        h_jitter = HandData(center_x=0.5, center_y=0.5, index_tip=(0.5005, 0.5005))
        self.engine.process(h_jitter)

        # Cursor position must remain exactly the same
        self.assertEqual(self.engine.smooth_cursor_x, initial_x)
        self.assertEqual(self.engine.smooth_cursor_y, initial_y)

    def test_air_mouse_disengage_on_fist(self):
        """Verify that making a Closed Fist disengages Air Mouse back to gesture mode."""
        self.engine.air_mouse_enabled = True
        self.engine.air_mouse_active = True

        h_fist = HandData(center_x=0.5, center_y=0.5, index_tip=(0.50, 0.50), gesture_name="Closed_Fist", gesture_score=0.90)
        self.engine.process(h_fist)

        self.assertFalse(self.engine.air_mouse_active)
        self.assertEqual(self.engine.current_state_text, "AIR MOUSE DISENGAGED")

    def test_double_clap_detection(self):
        """Verify that two rapid claps trigger DOUBLE_CLAP."""
        self.discrete_actions.clear()
        h_left = HandData(center_x=0.25, center_y=0.50)
        h_right = HandData(center_x=0.75, center_y=0.50)

        # 1. Hands open
        self.engine.process(h_left, all_hands=[h_left, h_right])
        time.sleep(0.04)

        # Clap 1: hands collide
        h_c1_l = HandData(center_x=0.48, center_y=0.50)
        h_c1_r = HandData(center_x=0.52, center_y=0.50)
        self.engine.process(h_c1_l, all_hands=[h_c1_l, h_c1_r])
        time.sleep(0.05)

        # Hands separate
        h_sep_l = HandData(center_x=0.35, center_y=0.50)
        h_sep_r = HandData(center_x=0.65, center_y=0.50)
        self.engine.process(h_sep_l, all_hands=[h_sep_l, h_sep_r])
        time.sleep(0.18)

        # Clap 2: hands collide again
        res = self.engine.process(h_c1_l, all_hands=[h_c1_l, h_c1_r])

        self.assertEqual(res, HandAction.DOUBLE_CLAP)
        self.assertTrue(any(a[0] == HandAction.DOUBLE_CLAP for a in self.discrete_actions))

    def test_finger_to_lips_quiet(self):
        """Verify that holding index finger up to lips triggers SHH_QUIET volume mute."""
        self.discrete_actions.clear()
        mouth_pos = (0.50, 0.45)

        # Index pointing up positioned at mouth (0.50, 0.44)
        h_shh = HandData(
            center_x=0.50,
            center_y=0.60,
            index_tip=(0.50, 0.44),
            gesture_name="Pointing_Up",
            gesture_score=0.92
        )

        self.engine.process(h_shh, face_mouth_pos=mouth_pos)
        time.sleep(0.38)
        res = self.engine.process(h_shh, face_mouth_pos=mouth_pos)

        self.assertEqual(res, HandAction.SHH_QUIET)
        self.assertTrue(any(a[0] == HandAction.SHH_QUIET for a in self.discrete_actions))

    def test_palm_to_fist_and_fist_to_palm_transitions(self):
        """Verify that transitioning Palm -> Fist minimizes, and Fist -> Palm restores."""
        self.discrete_actions.clear()

        # Step 1: Hold Open_Palm
        h_palm = HandData(center_x=0.5, center_y=0.5, gesture_name="Open_Palm", gesture_score=0.90)
        self.engine.process(h_palm)
        time.sleep(0.12)
        self.engine.process(h_palm)

        # Step 2: Clench to Closed_Fist within transition window
        time.sleep(0.10)
        h_fist = HandData(center_x=0.5, center_y=0.5, gesture_name="Closed_Fist", gesture_score=0.90)
        res_min = self.engine.process(h_fist)
        self.assertEqual(res_min, HandAction.PALM_TO_FIST)
        self.assertTrue(any(a[0] == HandAction.PALM_TO_FIST for a in self.discrete_actions))

        # Step 3: Wait past cooldown and transition back to Open_Palm
        time.sleep(0.90)
        self.engine.process(h_fist)
        time.sleep(0.10)
        res_restore = self.engine.process(h_palm)
        self.assertEqual(res_restore, HandAction.FIST_TO_PALM)
        self.assertTrue(any(a[0] == HandAction.FIST_TO_PALM for a in self.discrete_actions))

    def test_standalone_fist_and_pointing_up_suppressed(self):
        """Verify standalone Closed_Fist and Pointing_Up do not trigger any action on their own."""
        self.discrete_actions.clear()

        # Standalone Closed_Fist held without previous palm transition
        h_fist = HandData(center_x=0.5, center_y=0.5, gesture_name="Closed_Fist", gesture_score=0.95)
        for _ in range(5):
            res = self.engine.process(h_fist)
            self.assertIsNone(res, "Standalone fist must not fire any action")
            time.sleep(0.04)

        self.assertEqual(len(self.discrete_actions), 0, "No actions should be dispatched for standalone fist")

        # Standalone Pointing_Up held away from mouth
        h_point = HandData(center_x=0.5, center_y=0.5, index_tip=(0.5, 0.45), gesture_name="Pointing_Up", gesture_score=0.95)
        for _ in range(5):
            res = self.engine.process(h_point, face_mouth_pos=(0.8, 0.8))
            self.assertIsNone(res, "Standalone pointing up must not fire any action")
            time.sleep(0.04)

        self.assertEqual(len(self.discrete_actions), 0, "No actions should be dispatched for standalone pointing up")

    def test_palm_to_fist_no_double_input(self):
        """Strictly verify that Palm -> Fist triggers ONLY PALM_TO_FIST, suppressing OPEN_PALM and scroll."""
        self.discrete_actions.clear()
        self.continuous_actions.clear()

        # Step 1: Open palm enters
        h_palm = HandData(center_x=0.5, center_y=0.5, gesture_name="Open_Palm", gesture_score=0.90)
        self.engine.process(h_palm)
        time.sleep(0.08)
        self.engine.process(h_palm)

        # Step 2: Clench into fist
        time.sleep(0.08)
        h_fist = HandData(center_x=0.5, center_y=0.52, gesture_name="Closed_Fist", gesture_score=0.90)
        res = self.engine.process(h_fist)
        self.assertEqual(res, HandAction.PALM_TO_FIST)

        # Ensure NO OPEN_PALM was emitted and NO scroll was emitted
        discrete_names = [a[0] for a in self.discrete_actions]
        self.assertEqual(discrete_names, [HandAction.PALM_TO_FIST])
        self.assertEqual(len(self.continuous_actions), 0, "No continuous scroll during palm-to-fist clench")

        # Step 3: Continued holding of fist should not trigger any bounce or new action
        time.sleep(0.05)
        res_hold = self.engine.process(h_fist)
        self.assertIsNone(res_hold)
        self.assertEqual(len(self.discrete_actions), 1)

    def test_holo_expand_and_shrink(self):
        """Verify dual-hand holographic window expand (spread apart) and shrink (bring together)."""
        self.discrete_actions.clear()

        # Initial dual-hand position: 0.20 apart
        h1 = HandData(center_x=0.40, center_y=0.50, handedness="Left")
        h2 = HandData(center_x=0.60, center_y=0.50, handedness="Right")
        self.engine.process(h1, all_hands=[h1, h2])
        time.sleep(0.05)

        # Spread hands apart: distance increases to 0.44 (delta = +0.24, vel >= 0.25)
        h1_spread = HandData(center_x=0.28, center_y=0.50, handedness="Left")
        h2_spread = HandData(center_x=0.72, center_y=0.50, handedness="Right")
        res_expand = self.engine.process(h1_spread, all_hands=[h1_spread, h2_spread])
        self.assertEqual(res_expand, HandAction.HOLO_EXPAND)
        self.assertTrue(any(a[0] == HandAction.HOLO_EXPAND for a in self.discrete_actions))

        # Wait past cooldown
        time.sleep(0.35)
        self.discrete_actions.clear()

        # Seed again at wide distance 0.44
        self.engine.process(h1_spread, all_hands=[h1_spread, h2_spread])
        time.sleep(0.05)

        # Bring hands together: distance decreases to 0.20 (delta = -0.24)
        res_shrink = self.engine.process(h1, all_hands=[h1, h2])
        self.assertEqual(res_shrink, HandAction.HOLO_SHRINK)
        self.assertTrue(any(a[0] == HandAction.HOLO_SHRINK for a in self.discrete_actions))

    def test_dual_swipe_left_and_right(self):
        """Verify both hands sweeping horizontally in parallel switches desktops."""
        self.discrete_actions.clear()

        # Initial dual-hand positions
        h1 = HandData(center_x=0.60, center_y=0.50, handedness="Left")
        h2 = HandData(center_x=0.80, center_y=0.50, handedness="Right")
        self.engine.process(h1, all_hands=[h1, h2])
        time.sleep(0.05)

        # Sweep both hands quickly LEFT in parallel (delta x ~ -0.15 for both)
        h1_left = HandData(center_x=0.45, center_y=0.50, handedness="Left")
        h2_left = HandData(center_x=0.65, center_y=0.50, handedness="Right")
        res_left = self.engine.process(h1_left, all_hands=[h1_left, h2_left])
        self.assertEqual(res_left, HandAction.DUAL_SWIPE_LEFT)
        self.assertTrue(any(a[0] == HandAction.DUAL_SWIPE_LEFT for a in self.discrete_actions))

        # Wait past cooldown
        time.sleep(0.35)
        self.discrete_actions.clear()

        # Seed again
        self.engine.process(h1_left, all_hands=[h1_left, h2_left])
        time.sleep(0.05)

        # Sweep both hands quickly RIGHT in parallel
        h1_right = HandData(center_x=0.62, center_y=0.50, handedness="Left")
        h2_right = HandData(center_x=0.82, center_y=0.50, handedness="Right")
        res_right = self.engine.process(h1_right, all_hands=[h1_right, h2_right])
        self.assertEqual(res_right, HandAction.DUAL_SWIPE_RIGHT)
        self.assertTrue(any(a[0] == HandAction.DUAL_SWIPE_RIGHT for a in self.discrete_actions))

    def test_repulsor_push(self):
        """Verify fast open palm scale expansion triggers REPULSOR_PUSH."""
        self.discrete_actions.clear()

        # Initial palm at normal scale (wrist at (0.5, 0.60), MCP9 at (0.5, 0.45) -> scale 0.15)
        lm_norm = [(0.5, 0.5, 0.0)] * 21
        lm_norm[0] = (0.5, 0.60, 0.0)
        lm_norm[9] = (0.5, 0.45, 0.0)
        h_norm = HandData(center_x=0.5, center_y=0.5, gesture_name="Open_Palm", gesture_score=0.90, landmarks=lm_norm)
        self.engine.process(h_norm)
        time.sleep(0.05)

        # Thrust forward toward camera: palm scale jumps to 0.25 (delta = +0.10)
        lm_thrust = [(0.5, 0.5, 0.0)] * 21
        lm_thrust[0] = (0.5, 0.65, 0.0)
        lm_thrust[9] = (0.5, 0.40, 0.0)
        h_thrust = HandData(center_x=0.5, center_y=0.5, gesture_name="Open_Palm", gesture_score=0.90, landmarks=lm_thrust)
        res = self.engine.process(h_thrust)
        self.assertEqual(res, HandAction.REPULSOR_PUSH)
        self.assertTrue(any(a[0] == HandAction.REPULSOR_PUSH for a in self.discrete_actions))


class TestActionDispatcherMouse(unittest.TestCase):
    def setUp(self):
        self.dispatcher = ActionDispatcher()

    def test_scroll_and_zoom_dispatch(self):
        res1 = self.dispatcher.dispatch("SCROLL_TEST", "scroll_up", "3")
        self.assertTrue(res1)
        res2 = self.dispatcher.dispatch("ZOOM_TEST", "zoom_in", "")
        self.assertTrue(res2)

    def test_cursor_move_and_click(self):
        # Should execute safely without throwing exceptions
        self.dispatcher.move_cursor(500, 400)
        self.dispatcher.mouse_click("left")
        self.dispatcher.mouse_click("right")

    def test_window_management_dispatch(self):
        res1 = self.dispatcher.dispatch("PALM_TO_FIST", "minimize_window", "super+h")
        self.assertTrue(res1)
        res2 = self.dispatcher.dispatch("FIST_TO_PALM", "restore_window", "alt+Tab")
        self.assertTrue(res2)

    def test_desktop_switching_dispatch(self):
        res1 = self.dispatcher.dispatch("SWIPE_LEFT", "hotkey", "ctrl+alt+Left")
        self.assertTrue(res1)
        res2 = self.dispatcher.dispatch("SWIPE_RIGHT", "hotkey", "ctrl+alt+Right")
        self.assertTrue(res2)

    def test_window_resize_and_desktop_dispatch(self):
        """Verify window resize, show desktop, and desktop switch dispatches."""
        self.assertTrue(self.dispatcher.dispatch("HOLO_EXPAND", "resize_window_expand", "super+Up"))
        self.assertTrue(self.dispatcher.dispatch("HOLO_SHRINK", "resize_window_shrink", "super+Down"))
        self.assertTrue(self.dispatcher.dispatch("REPULSOR_PUSH", "show_desktop", "super+d"))
        self.assertTrue(self.dispatcher.dispatch("DUAL_SWIPE_LEFT", "switch_desktop_left", "ctrl+alt+Left"))
        self.assertTrue(self.dispatcher.dispatch("DUAL_SWIPE_RIGHT", "switch_desktop_right", "ctrl+alt+Right"))


class TestModeEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_mode_switch(self):
        res = self.client.post("/api/mode", json={"mode": "hands"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["mode"], "hands")

        res_eyes = self.client.post("/api/mode", json={"mode": "eyes"})
        self.assertEqual(res_eyes.status_code, 200)
        self.assertEqual(res_eyes.json()["mode"], "eyes")

        # Revert back to hands
        self.client.post("/api/mode", json={"mode": "hands"})

    def test_air_click_open_terminal(self):
        from server.app import state
        state.config.active = True
        state.config.mode = "hands"
        state.config.hand_mappings["AIR_CLICK"] = ActionConfig(
            type="command", value="gnome-terminal", description="Open Terminal", enabled=True
        )
        dispatched = []
        with unittest.mock.patch.object(state.dispatcher, 'dispatch', side_effect=lambda **kwargs: dispatched.append(kwargs)):
            state._handle_cursor_click("left")
            self.assertEqual(len(dispatched), 1)
            self.assertEqual(dispatched[0]["gesture"], "AIR_CLICK")
            self.assertEqual(dispatched[0]["action_type"], "command")
            self.assertEqual(dispatched[0]["action_value"], "gnome-terminal")

    def test_face_blur_endpoint(self):
        """Verify POST /api/blur-face toggles the blur_face config state."""
        # Toggle on
        res_on = self.client.post("/api/blur-face")
        self.assertEqual(res_on.status_code, 200)
        self.assertTrue(res_on.json()["blur_face"])

        # Toggle off
        res_off = self.client.post("/api/blur-face")
        self.assertEqual(res_off.status_code, 200)
        self.assertFalse(res_off.json()["blur_face"])

    def test_face_blur_filter(self):
        """Verify BlinkDetector.apply_face_blur anonymizes the facial region."""
        # Create a test image with high-entropy color gradient
        np.random.seed(42)
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        orig = img.copy()
        bbox = (40, 40, 160, 160)

        blurred = BlinkDetector.apply_face_blur(img, bbox)
        self.assertEqual(blurred.shape, orig.shape)
        # Inside the blurred region, the pixels must be altered by mosaic & blur
        self.assertFalse(np.array_equal(blurred, orig))

    def test_browsing_preset_profile_endpoint(self):
        """Verify applying the browsing preset profile through the API."""
        res = self.client.post("/api/preset", json={"preset": "browsing"})
        self.assertEqual(res.status_code, 200)
        mappings = res.json()["mappings"]
        self.assertIn("HOLO_EXPAND", mappings)
        self.assertIn("HOLO_SHRINK", mappings)
        self.assertIn("DUAL_SWIPE_LEFT", mappings)
        self.assertIn("DUAL_SWIPE_RIGHT", mappings)
        self.assertIn("REPULSOR_PUSH", mappings)
        self.assertTrue(mappings["HOLO_EXPAND"]["enabled"])
        self.assertTrue(mappings["HOLO_SHRINK"]["enabled"])
        self.assertFalse(mappings["OPEN_PALM"]["enabled"])


if __name__ == "__main__":
    unittest.main()
