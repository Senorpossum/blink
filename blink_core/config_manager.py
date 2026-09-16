"""
Configuration management for BlinkFlow & HandFlow.
Handles persistence, presets, and sensitivity thresholds for both Hands and Eyes mode.
"""

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class ActionConfig:
    type: str  # "hotkey", "command", "scroll_up", "scroll_down", "zoom_in", "zoom_out", "click"
    value: str  # e.g. "space", "ctrl+Tab", "3", "pactl set-sink-mute @DEFAULT_SINK@ toggle"
    description: str = ""
    enabled: bool = True


@dataclass
class EyeSensitivityConfig:
    blink_threshold: float = 0.55       # Blendshape score (0.0 open to 1.0 closed) to register closed eye
    wink_open_threshold: float = 0.35   # Other eye must be below this to be considered open
    min_blink_duration_ms: int = 70     # Below this is a glitch/flutter
    max_normal_blink_ms: int = 320      # Natural involuntary blink duration
    long_blink_duration_ms: int = 480   # Deliberate long blink threshold
    double_blink_window_ms: int = 450   # Max gap between 2 blinks
    triple_blink_window_ms: int = 700   # Max window for 3 blinks
    cooldown_ms: int = 400              # Suppression cooldown after any gesture action


# Backwards compatibility alias
SensitivityConfig = EyeSensitivityConfig


@dataclass
class HandSensitivityConfig:
    scroll_deadzone: float = 0.038      # Min hand movement to initiate scroll (3.8% screen)
    scroll_speed: int = 3               # Scroll wheel steps per tick
    scroll_inverted: bool = False       # Invert scroll direction (natural vs standard)
    input_delay_ms: int = 250           # Cooldown between consecutive gesture inputs (250ms)
    swipe_velocity_thresh: float = 0.45 # Min horizontal velocity for swipe trigger
    swipe_min_dist: float = 0.08        # Min horizontal distance (8% of screen)
    pose_hold_ms: int = 350             # Time pose must be held to activate (350ms)
    air_mouse_enabled: bool = True      # Air mouse fingertip cursor tracking
    air_mouse_idle_sec: float = 2.0     # Steady idle duration to engage air mouse (2.0s)
    dwell_click_ms: int = 700           # Dwell stationary duration to auto-click (700ms)
    dwell_click_enabled: bool = False   # Disabled by default so clicks only occur on fingertip bend
    # Legacy aliases
    scroll_interval_ms: int = 250
    zoom_delta_thresh: float = 0.070
    zoom_interval_ms: int = 240
    swipe_cooldown_ms: int = 250
    pose_cooldown_ms: int = 250


@dataclass
class AppConfig:
    mode: str = "hands"                 # "hands" or "eyes"
    active: bool = True
    camera_index: int = 0
    camera_fps: int = 30
    camera_width: int = 640
    camera_height: int = 480
    audio_feedback: bool = True
    audio_volume: float = 0.5
    draw_mesh: bool = True
    blur_face: bool = False             # Anonymity blur over face region
    sensitivity: EyeSensitivityConfig = field(default_factory=EyeSensitivityConfig)
    hand_sensitivity: HandSensitivityConfig = field(default_factory=HandSensitivityConfig)
    mappings: Dict[str, ActionConfig] = field(default_factory=dict)
    hand_mappings: Dict[str, ActionConfig] = field(default_factory=dict)

    def __post_init__(self):
        if not self.mappings:
            self.mappings = AppConfig.get_default_eye_mappings()
        if not self.hand_mappings:
            self.hand_mappings = AppConfig.get_default_hand_mappings()

    @staticmethod
    def get_default_eye_mappings() -> Dict[str, ActionConfig]:
        return {
            "DOUBLE_BLINK": ActionConfig(type="hotkey", value="Page_Down", description="Scroll Down", enabled=True),
            "TRIPLE_BLINK": ActionConfig(type="hotkey", value="Page_Up", description="Scroll Up", enabled=True),
            "LONG_BLINK": ActionConfig(type="hotkey", value="space", description="Play / Pause", enabled=True),
            "LEFT_WINK": ActionConfig(type="hotkey", value="ctrl+Shift+Tab", description="Previous Tab", enabled=True),
            "RIGHT_WINK": ActionConfig(type="hotkey", value="ctrl+Tab", description="Next Tab", enabled=True),
            "LONG_LEFT_WINK": ActionConfig(type="hotkey", value="ctrl+w", description="Close Tab", enabled=False),
            "LONG_RIGHT_WINK": ActionConfig(type="hotkey", value="ctrl+t", description="Open Tab", enabled=False),
        }

    @staticmethod
    def get_default_hand_mappings() -> Dict[str, ActionConfig]:
        return {
            "SCROLL_UP": ActionConfig(type="scroll_up", value="3", description="Scroll Page Up", enabled=True),
            "SCROLL_DOWN": ActionConfig(type="scroll_down", value="3", description="Scroll Page Down", enabled=True),
            "SWIPE_LEFT": ActionConfig(type="hotkey", value="ctrl+alt+Left", description="Switch Desktop Left", enabled=True),
            "SWIPE_RIGHT": ActionConfig(type="hotkey", value="ctrl+alt+Right", description="Switch Desktop Right", enabled=True),
            "OPEN_PALM": ActionConfig(type="hotkey", value="space", description="Open Palm: Play / Pause Media", enabled=True),
            "VICTORY": ActionConfig(type="hotkey", value="ctrl+t", description="Victory / Peace: New Tab", enabled=True),
            "THUMB_UP": ActionConfig(type="hotkey", value="XF86AudioRaiseVolume", description="Thumbs Up: Volume Up", enabled=True),
            "THUMB_DOWN": ActionConfig(type="hotkey", value="XF86AudioLowerVolume", description="Thumbs Down: Volume Down", enabled=True),
            "AIR_CLICK": ActionConfig(type="command", value="gnome-terminal", description="Air Click: Bend Fingertip / Tap Forward", enabled=True),
            "DOUBLE_CLAP": ActionConfig(type="command", value="xdg-open https://google.com", description="Double Clap: Open Browser", enabled=True),
            "SHH_QUIET": ActionConfig(type="hotkey", value="XF86AudioMute", description="Finger to Lips: Toggle Mute", enabled=True),
            "PALM_TO_FIST": ActionConfig(type="minimize_window", value="super+h", description="Hand into Fist: Close/Minimize Current Window", enabled=True),
            "FIST_TO_PALM": ActionConfig(type="restore_window", value="alt+Tab", description="Fist into Hand: Reopen Last Window", enabled=True),
            "HOLO_EXPAND": ActionConfig(type="resize_window_expand", value="super+Up", description="Spread Hands Apart: Maximize Window", enabled=True),
            "HOLO_SHRINK": ActionConfig(type="resize_window_shrink", value="super+Down", description="Bring Hands Together: Restore Window", enabled=True),
            "DUAL_SWIPE_LEFT": ActionConfig(type="switch_desktop_left", value="ctrl+alt+Left", description="Dual Hand Sweep Left: Switch Desktop Left", enabled=True),
            "DUAL_SWIPE_RIGHT": ActionConfig(type="switch_desktop_right", value="ctrl+alt+Right", description="Dual Hand Sweep Right: Switch Desktop Right", enabled=True),
            "REPULSOR_PUSH": ActionConfig(type="show_desktop", value="super+d", description="Palm Push Forward: Show Desktop", enabled=True),
        }

    @staticmethod
    def get_hand_preset_profiles() -> Dict[str, Dict[str, ActionConfig]]:
        return {
            "browsing": {
                "SCROLL_UP": ActionConfig(type="scroll_up", value="3", description="Scroll Up", enabled=True),
                "SCROLL_DOWN": ActionConfig(type="scroll_down", value="3", description="Scroll Down", enabled=True),
                "SWIPE_LEFT": ActionConfig(type="hotkey", value="ctrl+Tab", description="Next Tab", enabled=True),
                "SWIPE_RIGHT": ActionConfig(type="hotkey", value="ctrl+Shift+Tab", description="Previous Tab", enabled=True),
                "VICTORY": ActionConfig(type="hotkey", value="ctrl+t", description="New Tab", enabled=True),
                "PALM_TO_FIST": ActionConfig(type="minimize_window", value="super+h", description="Close Window", enabled=True),
                "FIST_TO_PALM": ActionConfig(type="restore_window", value="alt+Tab", description="Reopen Window", enabled=True),
                "HOLO_EXPAND": ActionConfig(type="resize_window_expand", value="super+Up", description="Maximize Window", enabled=True),
                "HOLO_SHRINK": ActionConfig(type="resize_window_shrink", value="super+Down", description="Restore Window", enabled=True),
                "DUAL_SWIPE_LEFT": ActionConfig(type="switch_desktop_left", value="ctrl+alt+Left", description="Switch Desktop Left", enabled=True),
                "DUAL_SWIPE_RIGHT": ActionConfig(type="switch_desktop_right", value="ctrl+alt+Right", description="Switch Desktop Right", enabled=True),
                "REPULSOR_PUSH": ActionConfig(type="show_desktop", value="super+d", description="Show Desktop", enabled=True),
                "OPEN_PALM": ActionConfig(type="hotkey", value="space", description="Play / Pause", enabled=False),
            },
            "media": {
                "OPEN_PALM": ActionConfig(type="hotkey", value="XF86AudioPlay", description="Play / Pause", enabled=True),
                "SHH_QUIET": ActionConfig(type="hotkey", value="XF86AudioMute", description="Finger to Lips: Toggle Mute", enabled=True),
                "THUMB_UP": ActionConfig(type="hotkey", value="XF86AudioRaiseVolume", description="Volume Up", enabled=True),
                "THUMB_DOWN": ActionConfig(type="hotkey", value="XF86AudioLowerVolume", description="Volume Down", enabled=True),
                "SWIPE_LEFT": ActionConfig(type="hotkey", value="XF86AudioNext", description="Next Track", enabled=True),
                "SWIPE_RIGHT": ActionConfig(type="hotkey", value="XF86AudioPrev", description="Previous Track", enabled=True),
                "PALM_TO_FIST": ActionConfig(type="minimize_window", value="super+h", description="Close Window", enabled=True),
                "FIST_TO_PALM": ActionConfig(type="restore_window", value="alt+Tab", description="Reopen Window", enabled=True),
                "SCROLL_UP": ActionConfig(type="scroll_up", value="2", description="Scroll Volume / Timeline", enabled=True),
                "SCROLL_DOWN": ActionConfig(type="scroll_down", value="2", description="Scroll Volume / Timeline", enabled=True),
            },
            "presentation": {
                "SWIPE_LEFT": ActionConfig(type="hotkey", value="Right", description="Next Slide", enabled=True),
                "SWIPE_RIGHT": ActionConfig(type="hotkey", value="Left", description="Previous Slide", enabled=True),
                "OPEN_PALM": ActionConfig(type="hotkey", value="b", description="Blank Screen (b)", enabled=True),
                "VICTORY": ActionConfig(type="hotkey", value="F5", description="Start Slideshow", enabled=True),
                "PALM_TO_FIST": ActionConfig(type="minimize_window", value="super+h", description="Close Window", enabled=True),
                "FIST_TO_PALM": ActionConfig(type="restore_window", value="alt+Tab", description="Reopen Window", enabled=True),
            },
        }


    @staticmethod
    def get_eye_preset_profiles() -> Dict[str, Dict[str, ActionConfig]]:
        return {
            "reading": {
                "DOUBLE_BLINK": ActionConfig(type="hotkey", value="Page_Down", description="Scroll Down", enabled=True),
                "TRIPLE_BLINK": ActionConfig(type="hotkey", value="Page_Up", description="Scroll Up", enabled=True),
                "LONG_BLINK": ActionConfig(type="hotkey", value="Home", description="Jump to Top", enabled=True),
                "LEFT_WINK": ActionConfig(type="hotkey", value="Up", description="Nudge Up", enabled=True),
                "RIGHT_WINK": ActionConfig(type="hotkey", value="Down", description="Nudge Down", enabled=True),
            },
            "browser": {
                "LEFT_WINK": ActionConfig(type="hotkey", value="ctrl+Shift+Tab", description="Previous Tab", enabled=True),
                "RIGHT_WINK": ActionConfig(type="hotkey", value="ctrl+Tab", description="Next Tab", enabled=True),
                "DOUBLE_BLINK": ActionConfig(type="hotkey", value="ctrl+t", description="New Tab", enabled=True),
                "LONG_BLINK": ActionConfig(type="hotkey", value="ctrl+w", description="Close Tab", enabled=True),
                "TRIPLE_BLINK": ActionConfig(type="hotkey", value="ctrl+r", description="Refresh Page", enabled=True),
            },
            "media": {
                "DOUBLE_BLINK": ActionConfig(type="hotkey", value="XF86AudioPlay", description="Play / Pause", enabled=True),
                "RIGHT_WINK": ActionConfig(type="hotkey", value="XF86AudioNext", description="Next Track", enabled=True),
                "LEFT_WINK": ActionConfig(type="hotkey", value="XF86AudioPrev", description="Previous Track", enabled=True),
                "LONG_BLINK": ActionConfig(type="hotkey", value="XF86AudioMute", description="Toggle Mute", enabled=True),
            },
        }

    # Alias for backwards compatibility
    get_preset_profiles = get_eye_preset_profiles


class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load()

    def load(self) -> AppConfig:
        if not os.path.exists(self.config_path):
            config = AppConfig()
            self.save(config)
            return config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            sens_data = data.get("sensitivity", {})
            sens_fields = {f.name for f in fields(EyeSensitivityConfig)}
            filtered_sens = {k: v for k, v in sens_data.items() if k in sens_fields}
            sens = EyeSensitivityConfig(**filtered_sens) if filtered_sens else EyeSensitivityConfig()

            h_sens_data = data.get("hand_sensitivity", {})
            h_sens_fields = {f.name for f in fields(HandSensitivityConfig)}
            filtered_h_sens = {k: v for k, v in h_sens_data.items() if k in h_sens_fields}
            h_sens = HandSensitivityConfig(**filtered_h_sens) if filtered_h_sens else HandSensitivityConfig()

            mappings_data = data.get("mappings", {})
            mappings = {}
            for k, v in mappings_data.items():
                if isinstance(v, dict):
                    mappings[k] = ActionConfig(**v)

            h_mappings_data = data.get("hand_mappings", {})
            h_mappings = {}
            for k, v in h_mappings_data.items():
                if isinstance(v, dict):
                    h_mappings[k] = ActionConfig(**v)

            config = AppConfig(
                mode=data.get("mode", "hands"),
                active=data.get("active", True),
                camera_index=data.get("camera_index", 0),
                camera_fps=data.get("camera_fps", 30),
                camera_width=data.get("camera_width", 640),
                camera_height=data.get("camera_height", 480),
                audio_feedback=data.get("audio_feedback", True),
                audio_volume=data.get("audio_volume", 0.5),
                draw_mesh=data.get("draw_mesh", True),
                blur_face=data.get("blur_face", False),
                sensitivity=sens,
                hand_sensitivity=h_sens,
                mappings=mappings or AppConfig.get_default_eye_mappings(),
                hand_mappings=h_mappings or AppConfig.get_default_hand_mappings(),
            )
            return config
        except Exception as e:
            print(f"[ConfigManager] Error loading {self.config_path}, falling back to defaults: {e}")
            config = AppConfig()
            self.save(config)
            return config

    def save(self, config: Optional[AppConfig] = None) -> None:
        if config is not None:
            self.config = config
        try:
            data = {
                "mode": self.config.mode,
                "active": self.config.active,
                "camera_index": self.config.camera_index,
                "camera_fps": self.config.camera_fps,
                "camera_width": self.config.camera_width,
                "camera_height": self.config.camera_height,
                "audio_feedback": self.config.audio_feedback,
                "audio_volume": self.config.audio_volume,
                "draw_mesh": self.config.draw_mesh,
                "blur_face": self.config.blur_face,
                "sensitivity": asdict(self.config.sensitivity),
                "hand_sensitivity": asdict(self.config.hand_sensitivity),
                "mappings": {k: asdict(v) for k, v in self.config.mappings.items()},
                "hand_mappings": {k: asdict(v) for k, v in self.config.hand_mappings.items()},
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ConfigManager] Error saving {self.config_path}: {e}")

    def update_from_dict(self, updates: Dict[str, Any]) -> AppConfig:
        if "mode" in updates:
            self.config.mode = str(updates["mode"])
        if "active" in updates:
            self.config.active = bool(updates["active"])
        if "audio_feedback" in updates:
            self.config.audio_feedback = bool(updates["audio_feedback"])
        if "draw_mesh" in updates:
            self.config.draw_mesh = bool(updates["draw_mesh"])
        if "blur_face" in updates:
            self.config.blur_face = bool(updates["blur_face"])
        if "camera_index" in updates:
            self.config.camera_index = int(updates["camera_index"])

        if "sensitivity" in updates and isinstance(updates["sensitivity"], dict):
            for k, v in updates["sensitivity"].items():
                if hasattr(self.config.sensitivity, k):
                    setattr(self.config.sensitivity, k, type(getattr(self.config.sensitivity, k))(v))

        if "hand_sensitivity" in updates and isinstance(updates["hand_sensitivity"], dict):
            for k, v in updates["hand_sensitivity"].items():
                if hasattr(self.config.hand_sensitivity, k):
                    setattr(self.config.hand_sensitivity, k, type(getattr(self.config.hand_sensitivity, k))(v))

        if "mappings" in updates and isinstance(updates["mappings"], dict):
            for k, v in updates["mappings"].items():
                if isinstance(v, dict):
                    self.config.mappings[k] = ActionConfig(
                        type=v.get("type", "hotkey"),
                        value=v.get("value", ""),
                        description=v.get("description", ""),
                        enabled=v.get("enabled", True),
                    )

        if "hand_mappings" in updates and isinstance(updates["hand_mappings"], dict):
            for k, v in updates["hand_mappings"].items():
                if isinstance(v, dict):
                    self.config.hand_mappings[k] = ActionConfig(
                        type=v.get("type", "hotkey"),
                        value=v.get("value", ""),
                        description=v.get("description", ""),
                        enabled=v.get("enabled", True),
                    )

        self.save()
        return self.config

    def apply_preset(self, preset_name: str) -> bool:
        hand_presets = AppConfig.get_hand_preset_profiles()
        eye_presets = AppConfig.get_eye_preset_profiles()

        applied = False

        # Match hand presets
        h_key = preset_name
        if h_key == "browser":
            h_key = "browsing"
        if h_key in hand_presets:
            self.config.hand_mappings.update(hand_presets[h_key])
            applied = True

        # Match eye presets
        e_key = preset_name
        if e_key == "browsing":
            e_key = "browser"
        if e_key in eye_presets:
            self.config.mappings.update(eye_presets[e_key])
            applied = True

        if applied:
            self.save()
            return True

        return False
