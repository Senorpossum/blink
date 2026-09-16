"""
Hand Gesture Engine for HandFlow.
Translates continuous hand movements and static poses into actions with:
- Global ~1 second delay/cooldown between any consecutive inputs (prevents runaway input)
- Zero pinch-to-zoom interference (pinch actions completely removed)
- Exponential Moving Average (EMA) position smoothing to eliminate webcam jitter
- Intentional displacement thresholds for scrolling (prevents micro-scrolls)
- Entry-suppression for swipes (no false triggers when bringing hand into view)
- Confidence and hold duration gating for static poses
"""

import logging
import math
import time
from collections import deque
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple

from .hand_detector import HandData

logger = logging.getLogger("HandGestureEngine")


class HandAction(Enum):
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"
    OPEN_PALM = "OPEN_PALM"
    CLOSED_FIST = "CLOSED_FIST"
    VICTORY = "VICTORY"
    POINTING_UP = "POINTING_UP"
    THUMB_UP = "THUMB_UP"
    THUMB_DOWN = "THUMB_DOWN"
    # New gestures
    AIR_CLICK = "AIR_CLICK"
    DOUBLE_CLAP = "DOUBLE_CLAP"
    SHH_QUIET = "SHH_QUIET"
    PALM_TO_FIST = "PALM_TO_FIST"
    FIST_TO_PALM = "FIST_TO_PALM"
    # Dual-Hand and Spatial Gestures
    HOLO_EXPAND = "HOLO_EXPAND"
    HOLO_SHRINK = "HOLO_SHRINK"
    DUAL_SWIPE_LEFT = "DUAL_SWIPE_LEFT"
    DUAL_SWIPE_RIGHT = "DUAL_SWIPE_RIGHT"
    REPULSOR_PUSH = "REPULSOR_PUSH"
    # Legacy enum values kept for config backwards compatibility
    ZOOM_IN = "ZOOM_IN"
    ZOOM_OUT = "ZOOM_OUT"
    PINCH_CLICK = "PINCH_CLICK"


class HandGestureEngine:
    def __init__(
        self,
        scroll_deadzone: float = 0.038,       # Min vertical displacement to trigger scroll (3.8% of screen)
        scroll_inverted: bool = False,        # Natural vs standard scroll
        swipe_velocity_thresh: float = 0.45,  # Speed required for swipe (natural flick)
        swipe_min_dist: float = 0.08,         # Min horizontal distance traveled (8% screen)
        pose_hold_ms: int = 350,              # Pose must be held steady for 350ms
        input_delay_ms: int = 250,            # 250ms delay between consecutive gesture inputs
        air_mouse_enabled: bool = True,       # Enable air mouse fingertip tracking
        air_mouse_idle_sec: float = 2.0,      # Steady idle time (seconds) to activate air mouse
        dwell_click_ms: int = 700,            # Hover duration to trigger dwell click
        dwell_click_enabled: bool = False,    # Disabled by default so clicks only trigger on finger bend
        screen_width: int = 1920,             # Monitor width for air mouse coordinate mapping
        screen_height: int = 1080,            # Monitor height for air mouse coordinate mapping
        # Legacy params kept for compatibility
        scroll_interval_ms: int = 250,
        zoom_delta_thresh: float = 0.070,
        zoom_interval_ms: int = 240,
        swipe_cooldown_ms: int = 250,
        pose_cooldown_ms: int = 250
    ):
        self.scroll_deadzone = scroll_deadzone
        self.scroll_inverted = scroll_inverted
        self.swipe_velocity_thresh = swipe_velocity_thresh
        self.swipe_min_dist = swipe_min_dist
        self.pose_hold_ms = pose_hold_ms
        self.input_delay_ms = input_delay_ms
        self.air_mouse_enabled = air_mouse_enabled
        self.air_mouse_idle_sec = air_mouse_idle_sec
        self.dwell_click_ms = dwell_click_ms
        self.dwell_click_enabled = dwell_click_enabled
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.on_continuous_action: Optional[Callable[[HandAction, float], None]] = None
        self.on_discrete_gesture: Optional[Callable[[HandAction, str], None]] = None
        self.on_cursor_move: Optional[Callable[[int, int], None]] = None
        self.on_cursor_click: Optional[Callable[[str], None]] = None

        # Smoothed tracking coordinates (EMA filter: alpha = 0.40)
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.anchor_y: Optional[float] = None

        # Tracking state
        self.prev_time: Optional[float] = None
        self.hand_entered_time: Optional[float] = None

        # Global input cooldown timer (enforces 250ms delay between inputs)
        self.last_action_time: float = 0.0

        # Pose hold tracker
        self.active_pose: Optional[str] = None
        self.pose_start_time: Optional[float] = None
        self.pose_triggered: bool = False

        # Air Mouse state
        self.air_mouse_active: bool = False
        self.idle_anchor: Optional[Tuple[float, float]] = None
        self.idle_start_time: Optional[float] = None
        self.idle_progress: float = 0.0
        self.cursor_screen_pos: Tuple[int, int] = (screen_width // 2, screen_height // 2)
        self.smooth_cursor_x: Optional[float] = None
        self.smooth_cursor_y: Optional[float] = None
        self.dwell_anchor: Optional[Tuple[int, int]] = None
        self.dwell_start_time: Optional[float] = None
        self.dwell_progress: float = 0.0
        self.dwell_clicked: bool = False
        self.last_air_right_click_time: float = 0.0

        # Precision Air Mouse enhancements
        self.deadband_px: float = 3.0
        self.cursor_freeze_until: float = 0.0
        self.last_forward_tap_time: float = 0.0
        self.last_middle_flick_time: float = 0.0
        self.last_click_event: Optional[str] = None
        self.last_click_time: float = 0.0
        self.finger_click_ready: bool = True
        self.depth_history: Deque[Tuple[float, float]] = deque(maxlen=8)
        self.middle_history: Deque[Tuple[float, float]] = deque(maxlen=8)

        # High-accuracy swipe trajectory history & cooldown
        self.swipe_history: Deque[Tuple[float, float, float]] = deque(maxlen=15)
        self.last_swipe_time: float = 0.0

        # Double Clap state
        self.clap_history: Deque[Tuple[float, float]] = deque(maxlen=8)
        self.last_single_clap_time: float = 0.0
        self.last_double_clap_time: float = 0.0
        self.clap_in_progress: bool = False

        # Finger to Lips ("Shh / Quiet") state
        self.shh_start_time: Optional[float] = None
        self.last_shh_time: float = 0.0

        # Pose sequence transition state (Palm <-> Fist) & Suppression
        self.last_stable_pose: Optional[str] = None
        self.last_stable_pose_time: float = 0.0
        self.last_transition_time: float = 0.0
        self.last_transition_direction: Optional[str] = None
        self.pose_suppress_until: float = 0.0

        # Dual-Hand state
        self.dual_hand_history: Deque[Tuple[float, float, float, float, float]] = deque(maxlen=15) # (dist, x1, y1, x2, t)
        self.last_dual_action_time: float = 0.0
        self.dual_action_feedback: Optional[str] = None
        self.dual_distance: float = 0.0

        # Palm Push state
        self.palm_scale_history: Deque[Tuple[float, float]] = deque(maxlen=10) # (scale, t)
        self.last_repulsor_time: float = 0.0

        # Position history for swipe velocity (last 6 frames)
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=6) # (x, y, t)

        self.current_state_text = "IDLE"

    def update_sensitivity(self, h_sens) -> None:
        self.scroll_deadzone = getattr(h_sens, "scroll_deadzone", self.scroll_deadzone)
        self.scroll_inverted = getattr(h_sens, "scroll_inverted", self.scroll_inverted)
        self.swipe_velocity_thresh = getattr(h_sens, "swipe_velocity_thresh", self.swipe_velocity_thresh)
        self.swipe_min_dist = getattr(h_sens, "swipe_min_dist", self.swipe_min_dist)
        self.pose_hold_ms = getattr(h_sens, "pose_hold_ms", self.pose_hold_ms)
        self.input_delay_ms = getattr(h_sens, "input_delay_ms", self.input_delay_ms)
        self.air_mouse_enabled = getattr(h_sens, "air_mouse_enabled", self.air_mouse_enabled)
        self.air_mouse_idle_sec = getattr(h_sens, "air_mouse_idle_sec", self.air_mouse_idle_sec)
        self.dwell_click_ms = getattr(h_sens, "dwell_click_ms", self.dwell_click_ms)
        self.dwell_click_enabled = getattr(h_sens, "dwell_click_enabled", self.dwell_click_enabled)

    def set_screen_resolution(self, width: int, height: int) -> None:
        self.screen_width = max(640, width)
        self.screen_height = max(480, height)

    def process(
        self,
        hand: Optional[HandData],
        all_hands: Optional[List[HandData]] = None,
        face_mouth_pos: Optional[Tuple[float, float]] = None
    ) -> Optional[HandAction]:
        now = time.time()

        if all_hands is None and hand is not None:
            all_hands = [hand]

        # -------------------------------------------------------------
        # 0. DOUBLE CLAP DETECTION (requires 2 hands in view)
        # -------------------------------------------------------------
        if all_hands and len(all_hands) >= 2:
            h1, h2 = all_hands[0], all_hands[1]
            dist_hands = math.hypot(h1.center_x - h2.center_x, h1.center_y - h2.center_y)
            self.clap_history.append((dist_hands, now))

            if len(self.clap_history) >= 2:
                approaching = False
                for old_d, old_t in reversed(list(self.clap_history)[:-1]):
                    dt = now - old_t
                    if 0.02 <= dt <= 0.35:
                        v_close = (dist_hands - old_d) / dt
                        if dist_hands < 0.16 and (v_close < -0.30 or dist_hands < 0.10):
                            approaching = True
                            break

                if approaching:
                    if not self.clap_in_progress and (now - self.last_single_clap_time) > 0.10:
                        self.clap_in_progress = True
                        if (now - self.last_single_clap_time) <= 1.20 and (now - self.last_double_clap_time) > 1.0:
                            self.last_double_clap_time = now
                            self.last_single_clap_time = 0.0
                            self.clap_history.clear()
                            self.current_state_text = "DOUBLE CLAP (OPEN BROWSER)"
                            self._fire_discrete(HandAction.DOUBLE_CLAP, "2 rapid claps")
                            return HandAction.DOUBLE_CLAP
                        else:
                            self.last_single_clap_time = now
                            self.current_state_text = "CLAP DETECTED (WAITING 2ND)"
                            return None
            if dist_hands > 0.22:
                self.clap_in_progress = False

        if hand is None:
            self._reset()
            self.current_state_text = "NO_HAND"
            return None

        # Track hand entry timestamp (suppress triggers during first 200ms of appearance)
        if self.hand_entered_time is None:
            self.hand_entered_time = now

        raw_x, raw_y = hand.center_x, hand.center_y
        itx, ity = hand.index_tip if (hand.index_tip and hand.index_tip != (0.0, 0.0)) else (raw_x, raw_y)
        pose = hand.gesture_name
        pose_score = getattr(hand, "gesture_score", 1.0)

        # Apply Exponential Moving Average (EMA) smoothing
        alpha = 0.40
        if self.smooth_x is None:
            self.smooth_x = raw_x
            self.smooth_y = raw_y
            self.anchor_y = raw_y
        else:
            self.smooth_x = alpha * raw_x + (1 - alpha) * self.smooth_x
            self.smooth_y = alpha * raw_y + (1 - alpha) * self.smooth_y

        hx, hy = self.smooth_x, self.smooth_y
        self.history.append((hx, hy, now))
        self.swipe_history.append((raw_x, raw_y, now))

        # Always track palm scale for repulsor push
        if len(hand.landmarks) >= 10:
            p_scale = math.hypot(hand.landmarks[9][0] - hand.landmarks[0][0], hand.landmarks[9][1] - hand.landmarks[0][1])
            self.palm_scale_history.append((p_scale, now))

        # Always track dual-hand metrics if 2 hands present
        if all_hands and len(all_hands) >= 2:
            h1 = all_hands[0]
            h2 = all_hands[1]
            left_h = h1 if h1.center_x <= h2.center_x else h2
            right_h = h2 if h1.center_x <= h2.center_x else h1
            dual_d = math.hypot(right_h.center_x - left_h.center_x, right_h.center_y - left_h.center_y)
            self.dual_distance = dual_d
            self.dual_hand_history.append((dual_d, left_h.center_x, right_h.center_x, left_h.center_y, now))
        else:
            self.dual_hand_history.clear()

        # -------------------------------------------------------------
        # AIR MOUSE TRACKING & IDLE ENGAGEMENT
        # -------------------------------------------------------------
        if self.air_mouse_enabled:
            if not self.air_mouse_active:
                # Accumulate steady idle duration
                if self.idle_anchor is None:
                    self.idle_anchor = (itx, ity)
                    self.idle_start_time = now
                    self.idle_progress = 0.0
                else:
                    dist = math.hypot(itx - self.idle_anchor[0], ity - self.idle_anchor[1])
                    if dist > 0.045:
                        # Hand moved significantly -> reset idle charge
                        self.idle_anchor = (itx, ity)
                        self.idle_start_time = now
                        self.idle_progress = 0.0
                    else:
                        elapsed_idle = now - self.idle_start_time
                        self.idle_progress = min(1.0, elapsed_idle / max(0.1, self.air_mouse_idle_sec))
                        if elapsed_idle >= self.air_mouse_idle_sec:
                            self.air_mouse_active = True
                            self.idle_progress = 1.0
                            self.current_state_text = "AIR MOUSE ACTIVE"
                            self.smooth_cursor_x = None
                            self.smooth_cursor_y = None
                            self.dwell_start_time = None
                            self.dwell_anchor = None
                            self.dwell_clicked = False
            else:
                # Air mouse is active! Check for disengagement (e.g. Closed Fist)
                if pose == "Closed_Fist" and pose_score >= 0.65:
                    self.air_mouse_active = False
                    self.idle_start_time = None
                    self.idle_anchor = None
                    self.idle_progress = 0.0
                    self.current_state_text = "AIR MOUSE DISENGAGED"
                    self.last_action_time = now
                    self.anchor_y = hy
                    self.prev_time = now
                    return None

                # 3D Depth and finger relative metrics
                index_rel_z = getattr(hand, "index_rel_z", 0.0)
                middle_dist = getattr(hand, "middle_dist", 0.0)
                if index_rel_z == 0.0 and len(hand.landmarks) >= 21:
                    index_rel_z = hand.landmarks[8][2] - hand.landmarks[5][2]
                if middle_dist == 0.0 and len(hand.landmarks) >= 21:
                    m_dx = hand.landmarks[12][0] - hand.landmarks[9][0]
                    m_dy = hand.landmarks[12][1] - hand.landmarks[9][1]
                    middle_dist = math.hypot(m_dx, m_dy)

                # Active screen coordinate mapping (Active Zone: 15% to 85% of webcam frame)
                nx = max(0.0, min(1.0, (itx - 0.15) / 0.70))
                ny = max(0.0, min(1.0, (ity - 0.15) / 0.70))
                target_sx = nx * self.screen_width
                target_sy = ny * self.screen_height

                # Initialize smoothed cursor position if first frame
                if self.smooth_cursor_x is None:
                    self.smooth_cursor_x = target_sx
                    self.smooth_cursor_y = target_sy

                # Check Tap-Freeze: if locked from a click/tap, prevent cursor drift
                if now < self.cursor_freeze_until:
                    cur_x = int(self.smooth_cursor_x)
                    cur_y = int(self.smooth_cursor_y)
                    self.cursor_screen_pos = (cur_x, cur_y)
                else:
                    d = math.hypot(target_sx - self.smooth_cursor_x, target_sy - self.smooth_cursor_y)
                    # Deadband jitter elimination: suppress micro-jitter under 3.0px
                    if d < self.deadband_px:
                        pass
                    else:
                        # Multi-tier precision smoothing:
                        # - Ultra precision (<12px): alpha = 0.12 (micro-targeting small UI elements)
                        # - Guided motion (<35px): alpha = 0.35 (comfortable smooth tracking)
                        # - Responsive transit (<80px): alpha = 0.65 (fast movement)
                        # - Snap/flick (>=80px): alpha = 0.85 (instant snap across screen)
                        if d < 12.0:
                            alpha_cur = 0.12
                        elif d < 35.0:
                            alpha_cur = 0.35
                        elif d < 80.0:
                            alpha_cur = 0.65
                        else:
                            alpha_cur = 0.85

                        self.smooth_cursor_x = alpha_cur * target_sx + (1.0 - alpha_cur) * self.smooth_cursor_x
                        self.smooth_cursor_y = alpha_cur * target_sy + (1.0 - alpha_cur) * self.smooth_cursor_y

                    cur_x = int(self.smooth_cursor_x)
                    cur_y = int(self.smooth_cursor_y)
                    self.cursor_screen_pos = (cur_x, cur_y)

                    if self.on_cursor_move:
                        try:
                            self.on_cursor_move(cur_x, cur_y)
                        except Exception as e:
                            logger.error(f"Error in on_cursor_move: {e}")

                # -------------------------------------------------------------
                # 1. FINGERTIP BEND TO LEFT CLICK (Instantaneous trigger, zero delay)
                # -------------------------------------------------------------
                index_bend = getattr(hand, "index_bend_ratio", 1.0)
                index_curl = getattr(hand, "index_curl_ratio", 1.0)
                if index_bend == 1.0 and len(hand.landmarks) >= 9:
                    p6, p7, p8 = hand.landmarks[6], hand.landmarks[7], hand.landmarks[8]
                    d8_6 = math.hypot(p8[0] - p6[0], p8[1] - p6[1], p8[2] - p6[2])
                    d_dip = math.hypot(p8[0] - p7[0], p8[1] - p7[1], p8[2] - p7[2]) + math.hypot(p7[0] - p6[0], p7[1] - p6[1], p7[2] - p6[2])
                    if d_dip > 1e-4:
                        index_bend = d8_6 / d_dip

                finger_bent = (index_bend < 0.78) or (index_curl < 0.76)
                finger_extended = (index_bend > 0.84) and (index_curl > 0.81)

                bend_clicked = False
                if finger_bent and self.finger_click_ready and (now - self.last_click_time) > 0.25:
                    bend_clicked = True
                    self.finger_click_ready = False
                    self.cursor_freeze_until = now + 0.150  # 150ms coordinate stabilization
                    self.last_click_event = "left"
                    self.last_click_time = now
                    self.current_state_text = f"FINGER BEND CLICK ({cur_x}, {cur_y})"
                    # Reset dwell tracking
                    self.dwell_anchor = (cur_x, cur_y)
                    self.dwell_start_time = now
                    self.dwell_progress = 0.0
                    self.dwell_clicked = True
                    if self.on_cursor_click:
                        try:
                            self.on_cursor_click("left")
                        except Exception as e:
                            logger.error(f"Error in finger bend on_cursor_click: {e}")
                elif finger_extended:
                    self.finger_click_ready = True

                # -------------------------------------------------------------
                # 2. FORWARD TAP TO LEFT CLICK (Secondary forward poke detector)
                # -------------------------------------------------------------
                self.depth_history.append((index_rel_z, now))
                forward_tapped = False
                if not bend_clicked and len(self.depth_history) >= 2 and (now - self.last_forward_tap_time) > 0.30 and (now - self.last_click_time) > 0.25:
                    for old_z, old_t in list(self.depth_history)[:-1]:
                        dt = now - old_t
                        if 0.030 <= dt <= 0.30:
                            dz = index_rel_z - old_z
                            vz = dz / dt
                            # Fast forward poke towards camera / screen
                            if vz <= -0.32 or (dz <= -0.035 and vz <= -0.28):
                                forward_tapped = True
                                break

                if forward_tapped:
                    self.last_forward_tap_time = now
                    self.cursor_freeze_until = now + 0.120  # 120ms tap-freeze
                    self.last_click_event = "left"
                    self.last_click_time = now
                    self.current_state_text = f"FORWARD TAP CLICK ({cur_x}, {cur_y})"
                    self.depth_history.clear()
                    # Reset dwell tracking
                    self.dwell_anchor = (cur_x, cur_y)
                    self.dwell_start_time = now
                    self.dwell_progress = 0.0
                    self.dwell_clicked = True
                    if self.on_cursor_click:
                        try:
                            self.on_cursor_click("left")
                        except Exception as e:
                            logger.error(f"Error in forward tap on_cursor_click: {e}")

                # -------------------------------------------------------------
                # 3. MIDDLE FINGER FLICK TO RIGHT CLICK
                # -------------------------------------------------------------
                self.middle_history.append((middle_dist, now))
                middle_flicked = False
                if len(self.middle_history) >= 2 and (now - self.last_middle_flick_time) > 0.40 and (now - self.last_air_right_click_time) > 0.40:
                    for old_m, old_t in list(self.middle_history)[:-1]:
                        dt = now - old_t
                        if 0.030 <= dt <= 0.30:
                            dm = middle_dist - old_m
                            vm = dm / dt
                            # Middle finger transitions from curled to extended rapidly
                            if (vm >= 0.40 and middle_dist >= 0.15) or (middle_dist >= 0.17 and old_m <= 0.12 and vm >= 0.30):
                                middle_flicked = True
                                break

                if middle_flicked:
                    self.last_middle_flick_time = now
                    self.last_air_right_click_time = now
                    self.cursor_freeze_until = now + 0.150  # 150ms tap-freeze
                    self.last_click_event = "right"
                    self.last_click_time = now
                    self.current_state_text = f"MIDDLE FLICK RIGHT CLICK ({cur_x}, {cur_y})"
                    self.middle_history.clear()
                    # Reset dwell tracking
                    self.dwell_anchor = (cur_x, cur_y)
                    self.dwell_start_time = now
                    self.dwell_progress = 0.0
                    self.dwell_clicked = True
                    if self.on_cursor_click:
                        try:
                            self.on_cursor_click("right")
                        except Exception as e:
                            logger.error(f"Error in middle flick right on_cursor_click: {e}")

                # -------------------------------------------------------------
                # 4. DWELL CLICK DETECTION (Optional; disabled by default)
                # -------------------------------------------------------------
                if self.dwell_click_enabled and not bend_clicked and not forward_tapped and not middle_flicked:
                    if self.dwell_anchor is None:
                        self.dwell_anchor = (cur_x, cur_y)
                        self.dwell_start_time = now
                        self.dwell_progress = 0.0
                        self.dwell_clicked = False
                    else:
                        dwell_dist = math.hypot(cur_x - self.dwell_anchor[0], cur_y - self.dwell_anchor[1])
                        if dwell_dist > 28.0:
                            self.dwell_anchor = (cur_x, cur_y)
                            self.dwell_start_time = now
                            self.dwell_progress = 0.0
                            self.dwell_clicked = False
                        else:
                            dwell_elapsed_ms = (now - self.dwell_start_time) * 1000.0
                            self.dwell_progress = min(1.0, dwell_elapsed_ms / max(1.0, float(self.dwell_click_ms)))
                            if dwell_elapsed_ms >= self.dwell_click_ms and not self.dwell_clicked:
                                self.dwell_clicked = True
                                self.last_click_event = "left"
                                self.last_click_time = now
                                self.cursor_freeze_until = now + 0.120
                                self.current_state_text = f"DWELL CLICK ({cur_x}, {cur_y})"
                                if self.on_cursor_click:
                                    try:
                                        self.on_cursor_click("left")
                                    except Exception as e:
                                        logger.error(f"Error in on_cursor_click: {e}")

                # -------------------------------------------------------------
                # 4. VICTORY / PEACE SIGN RIGHT CLICK (Secondary gesture)
                # -------------------------------------------------------------
                if pose == "Victory" and pose_score >= 0.65:
                    if (now - self.last_air_right_click_time) > 1.2:
                        self.last_air_right_click_time = now
                        self.last_click_event = "right"
                        self.last_click_time = now
                        self.cursor_freeze_until = now + 0.150
                        self.current_state_text = "RIGHT CLICK (PEACE)"
                        if self.on_cursor_click:
                            try:
                                self.on_cursor_click("right")
                            except Exception as e:
                                logger.error(f"Error in right on_cursor_click: {e}")

                if not self.dwell_clicked and not forward_tapped and not middle_flicked and (now - self.last_air_right_click_time) > 0.5 and (now - self.last_forward_tap_time) > 0.5:
                    if self.dwell_progress > 0.15:
                        self.current_state_text = f"AIR MOUSE [DWELL {int(self.dwell_progress * 100)}%] ({cur_x}, {cur_y})"
                    else:
                        self.current_state_text = f"AIR MOUSE ({cur_x}, {cur_y})"

                self.prev_time = now
                self.anchor_y = hy
                return None

        # Check global input delay (250ms between consecutive actions)
        cooldown_duration = self.input_delay_ms / 1000.0
        elapsed_since_action = now - self.last_action_time
        if elapsed_since_action < cooldown_duration:
            remaining_ms = int((cooldown_duration - elapsed_since_action) * 1000)
            self.current_state_text = f"DELAY ({remaining_ms}ms)"
            self.anchor_y = hy  # keep anchor moving with hand during delay
            self.prev_time = now
            return None

        # Track static pose state (with transition suppression lockout)
        if now < self.pose_suppress_until:
            self.active_pose = None
            self.pose_start_time = None
            self.pose_triggered = True
        elif pose != "None" and pose_score >= 0.65:
            if pose != self.active_pose:
                self.active_pose = pose
                self.pose_start_time = now
                self.pose_triggered = False
        else:
            self.active_pose = None
            self.pose_start_time = None
            self.pose_triggered = False

        # -------------------------------------------------------------
        # A. FINGER TO LIPS ("SHH / BE QUIET") - VOLUME MUTE TOGGLE
        # -------------------------------------------------------------
        is_pointing_up = (pose == "Pointing_Up" and pose_score >= 0.60) or (ity < hand.center_y - 0.08)
        at_mouth = False
        if face_mouth_pos is not None:
            dist_mouth = math.hypot(itx - face_mouth_pos[0], ity - face_mouth_pos[1])
            at_mouth = (dist_mouth < 0.14)
        else:
            at_mouth = (0.28 <= itx <= 0.72 and 0.20 <= ity <= 0.68)

        if is_pointing_up and at_mouth:
            if self.shh_start_time is None:
                self.shh_start_time = now
            else:
                shh_hold = now - self.shh_start_time
                if shh_hold >= 0.35 and (now - self.last_shh_time) > 1.20:
                    self.last_shh_time = now
                    self.shh_start_time = None
                    self.anchor_y = hy
                    self.current_state_text = "SHH / QUIET (VOLUME MUTE)"
                    self._fire_discrete(HandAction.SHH_QUIET, "Finger at lips for 350ms")
                    self.prev_time = now
                    return HandAction.SHH_QUIET
        else:
            self.shh_start_time = None

        if self.prev_time is None:
            self.prev_time = now
            self.anchor_y = hy
            return None

        # Suppress initial entry motion
        if (now - self.hand_entered_time) < 0.20:
            self.anchor_y = hy
            self.prev_time = now
            return None

        # -------------------------------------------------------------
        # B. POSE SEQUENCES: PALM TO FIST (CLOSE WINDOW) & FIST TO PALM (REOPEN WINDOW)
        # -------------------------------------------------------------
        if pose in ("Open_Palm", "Closed_Fist") and pose_score >= 0.60:
            if self.last_stable_pose is not None and self.last_stable_pose != pose:
                dt_trans = now - self.last_stable_pose_time
                if dt_trans <= 1.20 and (now - self.last_transition_time) > 0.40:
                    if self.last_stable_pose == "Open_Palm" and pose == "Closed_Fist":
                        if self.last_transition_direction != "PALM_TO_FIST" or (now - self.last_transition_time) > 0.65:
                            self.last_transition_time = now
                            self.last_transition_direction = "PALM_TO_FIST"
                            self.pose_suppress_until = now + 1.20  # Completely lock out OPEN_PALM double trigger!
                            self.active_pose = None
                            self.pose_start_time = None
                            self.pose_triggered = True
                            self.last_stable_pose = "Closed_Fist"
                            self.last_stable_pose_time = now
                            self.anchor_y = hy  # Reset scroll anchor to prevent motion scroll
                            self.current_state_text = "CLOSE WINDOW (PALM ➔ FIST)"
                            self._fire_discrete(HandAction.PALM_TO_FIST, f"Transition in {dt_trans:.2f}s")
                            self.prev_time = now
                            return HandAction.PALM_TO_FIST
                    elif self.last_stable_pose == "Closed_Fist" and pose == "Open_Palm":
                        if self.last_transition_direction != "FIST_TO_PALM" or (now - self.last_transition_time) > 0.65:
                            self.last_transition_time = now
                            self.last_transition_direction = "FIST_TO_PALM"
                            self.pose_suppress_until = now + 1.20  # Completely lock out OPEN_PALM double trigger!
                            self.active_pose = None
                            self.pose_start_time = None
                            self.pose_triggered = True
                            self.last_stable_pose = "Open_Palm"
                            self.last_stable_pose_time = now
                            self.anchor_y = hy  # Reset scroll anchor to prevent motion scroll
                            self.current_state_text = "REOPEN WINDOW (FIST ➔ PALM)"
                            self._fire_discrete(HandAction.FIST_TO_PALM, f"Transition in {dt_trans:.2f}s")
                            self.prev_time = now
                            return HandAction.FIST_TO_PALM

            self.last_stable_pose = pose
            self.last_stable_pose_time = now

        # -------------------------------------------------------------
        # C. PALM PUSH (Thrust Open Palm Forward)
        # -------------------------------------------------------------
        if (
            pose == "Open_Palm"
            and pose_score >= 0.65
            and (now - self.last_repulsor_time) > 0.80
            and (now - self.last_transition_time) > 0.50
            and len(self.palm_scale_history) >= 2
        ):
            p_scale = self.palm_scale_history[-1][0]
            for past_scale, past_t in list(self.palm_scale_history)[:-1]:
                dt_rep = now - past_t
                if 0.02 <= dt_rep <= 0.35:
                    d_scale = p_scale - past_scale
                    v_scale = d_scale / dt_rep
                    if d_scale >= 0.038 and v_scale >= 0.20:
                        self.last_repulsor_time = now
                        self.palm_scale_history.clear()
                        self.anchor_y = hy
                        self.pose_suppress_until = now + 0.90
                        self.current_state_text = "REPULSOR PUSH (SHOW DESKTOP)"
                        self._fire_discrete(HandAction.REPULSOR_PUSH, f"Thrust scale +{d_scale:.2f}")
                        self.prev_time = now
                        return HandAction.REPULSOR_PUSH

        # =============================================================
        # DUAL-HAND HOLOGRAPHIC INTERFACE (Two Hands)
        # =============================================================
        if all_hands and len(all_hands) >= 2:
            h1 = all_hands[0]
            h2 = all_hands[1]
            left_h = h1 if h1.center_x <= h2.center_x else h2
            right_h = h2 if h1.center_x <= h2.center_x else h1
            dual_d = self.dual_distance

            if len(self.dual_hand_history) >= 2 and (now - self.last_dual_action_time) > 0.35:
                # 1. Check Dual-Hand Horizontal Swipe (Throw Desktop)
                for past_d, past_lx, past_rx, past_ly, past_t in reversed(list(self.dual_hand_history)[:-1]):
                    dt_dual = now - past_t
                    if 0.02 <= dt_dual <= 0.45:
                        dlx = left_h.center_x - past_lx
                        drx = right_h.center_x - past_rx
                        # Both hands moved significantly in the same direction
                        if (dlx > 0.05 and drx > 0.05) or (dlx < -0.05 and drx < -0.05):
                            avg_vx = ((dlx + drx) / 2.0) / dt_dual
                            if abs(avg_vx) >= 0.35:
                                dual_swipe = HandAction.DUAL_SWIPE_RIGHT if avg_vx > 0 else HandAction.DUAL_SWIPE_LEFT
                                self.last_dual_action_time = now
                                self.dual_hand_history.clear()
                                self.anchor_y = hy
                                self.dual_action_feedback = f"{dual_swipe.value}"
                                self.current_state_text = f"DUAL SWIPE {dual_swipe.value} (VEL {abs(avg_vx):.2f})"
                                self._fire_discrete(dual_swipe, f"Dual Swipe Vel {avg_vx:.2f}")
                                self.prev_time = now
                                return dual_swipe

                # 2. Check Holo-Expand & Holo-Shrink (Resize Window)
                for past_d, past_lx, past_rx, past_ly, past_t in reversed(list(self.dual_hand_history)[:-1]):
                    dt_dual = now - past_t
                    if 0.02 <= dt_dual <= 0.45:
                        delta_d = dual_d - past_d
                        v_dist = delta_d / dt_dual
                        if delta_d >= 0.065 and v_dist >= 0.22:
                            # Hands spreading apart -> HOLO EXPAND (Maximize / Expand)
                            self.last_dual_action_time = now
                            self.dual_hand_history.clear()
                            self.anchor_y = hy
                            self.dual_action_feedback = "HOLO_EXPAND"
                            self.current_state_text = f"WINDOW EXPAND (d={dual_d:.2f}, +{delta_d:.2f})"
                            self._fire_discrete(HandAction.HOLO_EXPAND, f"Spread +{delta_d:.2f} (Vel {v_dist:.2f})")
                            self.prev_time = now
                            return HandAction.HOLO_EXPAND
                        elif delta_d <= -0.065 and v_dist <= -0.22:
                            # Hands moving together -> HOLO SHRINK (Restore / Shrink)
                            self.last_dual_action_time = now
                            self.dual_hand_history.clear()
                            self.anchor_y = hy
                            self.dual_action_feedback = "HOLO_SHRINK"
                            self.current_state_text = f"WINDOW SHRINK (d={dual_d:.2f}, {delta_d:.2f})"
                            self._fire_discrete(HandAction.HOLO_SHRINK, f"Contract {delta_d:.2f} (Vel {v_dist:.2f})")
                            self.prev_time = now
                            return HandAction.HOLO_SHRINK

        triggered_action: Optional[HandAction] = None

        # -------------------------------------------------------------
        # 1. HIGH-ACCURACY HORIZONTAL SWIPE GESTURES
        # -------------------------------------------------------------
        if not (all_hands and len(all_hands) >= 2) and len(self.swipe_history) >= 2 and (now - self.last_swipe_time) > 0.35:
            for past_x, past_y, past_t in reversed(list(self.swipe_history)[:-1]):
                dt_swipe = now - past_t
                if 0.05 <= dt_swipe <= 0.35:
                    dx = raw_x - past_x
                    dy = raw_y - past_y
                    vx = dx / dt_swipe

                    # Must satisfy horizontal dominance, minimum displacement, and velocity
                    if abs(dx) >= self.swipe_min_dist and abs(dx) > 1.30 * max(0.01, abs(dy)):
                        if abs(vx) >= self.swipe_velocity_thresh:
                            swipe_action = HandAction.SWIPE_RIGHT if dx > 0 else HandAction.SWIPE_LEFT
                            self.last_swipe_time = now
                            self.swipe_history.clear()
                            self.history.clear()
                            self.anchor_y = hy  # Reset scroll anchor to prevent any scroll trigger
                            self.current_state_text = f"{swipe_action.value} (VEL {abs(vx):.2f})"
                            self._fire_discrete(swipe_action, f"Displacement {dx:.2f}, Velocity {vx:.2f}")
                            self.prev_time = now
                            return swipe_action

        # -------------------------------------------------------------
        # 2. VERTICAL MOTION / INTENTIONAL SCROLLING
        # -------------------------------------------------------------
        if self.anchor_y is None:
            self.anchor_y = hy

        dy = hy - self.anchor_y

        # Do not scroll if a transition, dual-hand action, repulsor push, or swipe occurred recently
        scroll_suppressed = (
            (all_hands and len(all_hands) >= 2)
            or (now - self.last_swipe_time) <= 0.30
            or (now - self.last_transition_time) <= 0.45
            or (now - self.last_dual_action_time) <= 0.40
            or (now - self.last_repulsor_time) <= 0.40
        )
        if not scroll_suppressed:
            if abs(dy) >= self.scroll_deadzone:
                steps = max(2, min(5, int(abs(dy) * 60)))
                if dy < -self.scroll_deadzone:
                    # Hand moved UP -> Scroll UP
                    action = HandAction.SCROLL_UP if not self.scroll_inverted else HandAction.SCROLL_DOWN
                    self.current_state_text = f"SCROLL UP ({steps})"
                    self.anchor_y = hy
                    self._fire_continuous(action, float(steps))
                    triggered_action = action
                    self.prev_time = now
                    return triggered_action
                elif dy > self.scroll_deadzone:
                    # Hand moved DOWN -> Scroll DOWN
                    action = HandAction.SCROLL_DOWN if not self.scroll_inverted else HandAction.SCROLL_UP
                    self.current_state_text = f"SCROLL DOWN ({steps})"
                    self.anchor_y = hy
                    self._fire_continuous(action, float(steps))
                    triggered_action = action
                    self.prev_time = now
                    return triggered_action
            else:
                # Anchor smoothly tracks resting hand drift
                self.anchor_y = 0.15 * hy + 0.85 * self.anchor_y

        # -------------------------------------------------------------
        # 3. STATIC HAND POSES (Open_Palm, Victory, etc. - Suppressed during/after transitions)
        # -------------------------------------------------------------
        if (
            now >= self.pose_suppress_until
            and self.active_pose
            and not self.pose_triggered
            and self.pose_start_time
        ):
            hold_duration = (now - self.pose_start_time) * 1000.0
            if hold_duration >= self.pose_hold_ms:
                self.pose_triggered = True

                pose_action_map = {
                    "Open_Palm": HandAction.OPEN_PALM,
                    "Victory": HandAction.VICTORY,
                    "Thumb_Up": HandAction.THUMB_UP,
                    "Thumb_Down": HandAction.THUMB_DOWN,
                }
                if self.active_pose in pose_action_map:
                    matched_action = pose_action_map[self.active_pose]
                    self.current_state_text = f"POSE: {self.active_pose}"
                    self._fire_discrete(matched_action, f"Held {int(hold_duration)}ms")
                    triggered_action = matched_action
                    self.prev_time = now
                    return triggered_action

        if triggered_action is None:
            if self.air_mouse_enabled and self.idle_progress > 0.10:
                self.current_state_text = f"CHARGING AIR MOUSE ({int(self.idle_progress * 100)}%)"
            elif all_hands and len(all_hands) >= 2:
                self.current_state_text = f"DUAL-HAND LINK (d={self.dual_distance:.2f})"
            else:
                p_text = self.active_pose if self.active_pose else "READY"
                self.current_state_text = f"IDLE ({p_text})"

        self.prev_time = now
        return None

    def _fire_continuous(self, action: HandAction, intensity: float):
        self.last_action_time = time.time()
        logger.info(f"Hand continuous action: {action.value} ({intensity:.1f})")
        if self.on_continuous_action:
            try:
                self.on_continuous_action(action, intensity)
            except Exception as e:
                logger.error(f"Error in on_continuous_action: {e}")

    def _fire_discrete(self, action: HandAction, meta: str):
        self.last_action_time = time.time()
        logger.info(f"Hand gesture recognized: {action.value} ({meta})")
        if self.on_discrete_gesture:
            try:
                self.on_discrete_gesture(action, meta)
            except Exception as e:
                logger.error(f"Error in on_discrete_gesture: {e}")

    def _reset(self):
        self.smooth_x = None
        self.smooth_y = None
        self.anchor_y = None
        self.prev_time = None
        self.hand_entered_time = None
        self.active_pose = None
        self.pose_start_time = None
        self.pose_triggered = False
        self.history.clear()
        self.air_mouse_active = False
        self.idle_anchor = None
        self.idle_start_time = None
        self.idle_progress = 0.0
        self.smooth_cursor_x = None
        self.smooth_cursor_y = None
        self.dwell_anchor = None
        self.dwell_start_time = None
        self.dwell_progress = 0.0
        self.dwell_clicked = False
        self.cursor_freeze_until = 0.0
        self.last_click_event = None
        self.finger_click_ready = True
        self.depth_history.clear()
        self.middle_history.clear()
        self.swipe_history.clear()
        self.clap_history.clear()
        self.clap_in_progress = False
        self.shh_start_time = None
        self.last_stable_pose = None
        self.last_stable_pose_time = 0.0
        self.last_transition_direction = None
        self.pose_suppress_until = 0.0
        self.dual_hand_history.clear()
        self.last_dual_action_time = 0.0
        self.dual_action_feedback = None
        self.dual_distance = 0.0
        self.palm_scale_history.clear()
        self.last_repulsor_time = 0.0
