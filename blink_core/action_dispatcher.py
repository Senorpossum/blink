"""
Action dispatcher for BlinkFlow & HandFlow.
Executes mapped keyboard shortcuts, mouse actions (scroll, zoom, click), or shell commands asynchronously.
"""

import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("ActionDispatcher")
logging.basicConfig(level=logging.INFO)

# Try importing pynput controllers
try:
    from pynput.keyboard import Controller as PynputKeyboard, Key
    from pynput.mouse import Button, Controller as PynputMouse
    PYNPUT_AVAILABLE = True
    pynput_kb = PynputKeyboard()
    pynput_mouse = PynputMouse()
except Exception as e:
    PYNPUT_AVAILABLE = False
    pynput_kb = None
    pynput_mouse = None
    logger.warning(f"pynput not available or failed to initialize: {e}")


class ActionDispatcher:
    def __init__(self, max_history: int = 50):
        self.xdotool_path = shutil.which("xdotool")
        self.history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.on_action_dispatched: Optional[Callable[[Dict[str, Any]], None]] = None
        self._lock = threading.Lock()
        self.screen_width, self.screen_height = self.get_screen_resolution()
        self.last_minimized_window: Optional[str] = None

    def get_screen_resolution(self) -> Tuple[int, int]:
        """
        Detects primary screen width and height. Defaults to 1920x1080.
        """
        try:
            if self.xdotool_path:
                out = subprocess.check_output([self.xdotool_path, "getdisplaygeometry"], stderr=subprocess.DEVNULL, timeout=1).decode().strip()
                parts = out.split()
                if len(parts) >= 2:
                    return int(parts[0]), int(parts[1])
        except Exception:
            pass

        try:
            out = subprocess.check_output(["xrandr"], stderr=subprocess.DEVNULL, timeout=1).decode()
            import re
            m = re.search(r'current (\d+) x (\d+)', out)
            if m:
                return int(m.group(1)), int(m.group(2))
        except Exception:
            pass

        return 1920, 1080

    def move_cursor(self, screen_x: int, screen_y: int):
        """
        Moves the OS mouse pointer to (screen_x, screen_y).
        """
        try:
            if PYNPUT_AVAILABLE and pynput_mouse:
                pynput_mouse.position = (int(screen_x), int(screen_y))
            elif self.xdotool_path:
                subprocess.run([self.xdotool_path, "mousemove", str(int(screen_x)), str(int(screen_y))], capture_output=True, timeout=0.1)
        except Exception as e:
            logger.debug(f"Cursor move error: {e}")

    def mouse_click(self, button: str = "left"):
        """
        Simulates an OS mouse click ('left' or 'right').
        """
        def _click():
            try:
                if PYNPUT_AVAILABLE and pynput_mouse:
                    btn = Button.right if "right" in button.lower() else Button.left
                    pynput_mouse.click(btn)
                elif self.xdotool_path:
                    btn_num = "3" if "right" in button.lower() else "1"
                    subprocess.run([self.xdotool_path, "click", btn_num], capture_output=True, timeout=1)
            except Exception as e:
                logger.error(f"Error executing mouse click: {e}")

        threading.Thread(target=_click, daemon=True).start()

    def dispatch(self, gesture: str, action_type: str, action_value: str, description: str = "") -> bool:
        """
        Dispatches an action in a background thread to prevent blocking the vision loop.
        """
        if not action_value and action_type not in (
            "scroll_up", "scroll_down", "zoom_in", "zoom_out", "click",
            "minimize_window", "restore_window", "resize_window_expand",
            "resize_window_shrink", "show_desktop", "switch_desktop_left", "switch_desktop_right"
        ):
            return False

        threading.Thread(
            target=self._execute_sync,
            args=(gesture, action_type, action_value, description),
            daemon=True
        ).start()
        return True

    def scroll(self, direction: str, amount: int = 2):
        """
        Smoothly scrolls the screen up or down.
        """
        threading.Thread(
            target=self._execute_scroll,
            args=(direction, amount),
            daemon=True
        ).start()

    def zoom(self, direction: str):
        """
        Zooms in or out.
        """
        threading.Thread(
            target=self._execute_zoom,
            args=(direction,),
            daemon=True
        ).start()

    def minimize_active_window(self, fallback_hotkey: str = "super+h") -> Tuple[bool, Optional[str]]:
        """
        Minimizes currently focused window and remembers its ID for restoration.
        """
        try:
            if self.xdotool_path:
                try:
                    wid = subprocess.check_output([self.xdotool_path, "getactivewindow"], stderr=subprocess.DEVNULL, timeout=0.5).decode().strip()
                    if wid:
                        self.last_minimized_window = wid
                        subprocess.run([self.xdotool_path, "windowminimize", wid], capture_output=True, timeout=0.5)
                        return True, None
                except Exception:
                    pass
            return self._send_hotkey(fallback_hotkey or "super+h")
        except Exception as e:
            return False, str(e)

    def restore_last_window(self, fallback_hotkey: str = "alt+Tab") -> Tuple[bool, Optional[str]]:
        """
        Restores / unminimizes the last minimized window or switches back to last active window.
        """
        try:
            if self.last_minimized_window and self.xdotool_path:
                try:
                    wid = self.last_minimized_window
                    subprocess.run([self.xdotool_path, "windowactivate", wid], capture_output=True, timeout=0.5)
                    self.last_minimized_window = None
                    return True, None
                except Exception:
                    pass
            return self._send_hotkey(fallback_hotkey or "alt+Tab")
        except Exception as e:
            return False, str(e)

    def resize_window_expand(self, fallback_hotkey: str = "super+Up") -> Tuple[bool, Optional[str]]:
        """
        Dual-Hand Expand: Maximizes active window or expands width/height.
        """
        try:
            return self._send_hotkey(fallback_hotkey or "super+Up")
        except Exception as e:
            return False, str(e)

    def resize_window_shrink(self, fallback_hotkey: str = "super+Down") -> Tuple[bool, Optional[str]]:
        """
        Dual-Hand Shrink: Restores / unmaximizes or shrinks active window.
        """
        try:
            return self._send_hotkey(fallback_hotkey or "super+Down")
        except Exception as e:
            return False, str(e)

    def show_desktop(self, fallback_hotkey: str = "super+d") -> Tuple[bool, Optional[str]]:
        """
        Palm Push: Toggles Show Desktop / Overview.
        """
        try:
            return self._send_hotkey(fallback_hotkey or "super+d")
        except Exception as e:
            return False, str(e)

    def switch_desktop_left(self, fallback_hotkey: str = "ctrl+alt+Left") -> Tuple[bool, Optional[str]]:
        """
        Dual-Hand Desktop Throw Left.
        """
        try:
            return self._send_hotkey(fallback_hotkey or "ctrl+alt+Left")
        except Exception as e:
            return False, str(e)

    def switch_desktop_right(self, fallback_hotkey: str = "ctrl+alt+Right") -> Tuple[bool, Optional[str]]:
        """
        Dual-Hand Desktop Throw Right.
        """
        try:
            return self._send_hotkey(fallback_hotkey or "ctrl+alt+Right")
        except Exception as e:
            return False, str(e)

    def _execute_scroll(self, direction: str, amount: int = 2):
        try:
            steps = max(1, min(10, amount))
            if PYNPUT_AVAILABLE and pynput_mouse:
                dy = steps if direction == "up" else -steps
                pynput_mouse.scroll(0, dy)
            elif self.xdotool_path:
                button = "4" if direction == "up" else "5"
                cmd = [self.xdotool_path, "click", "--repeat", str(steps), button]
                subprocess.run(cmd, capture_output=True, timeout=1)
        except Exception as e:
            logger.error(f"Error during scroll: {e}")

    def _execute_zoom(self, direction: str):
        try:
            if direction == "in":
                self._send_hotkey("ctrl+plus")
            else:
                self._send_hotkey("ctrl+minus")
        except Exception as e:
            logger.error(f"Error during zoom: {e}")

    def _execute_sync(self, gesture: str, action_type: str, action_value: str, description: str = "") -> bool:
        start_time = time.time()
        success = False
        error_msg = None

        try:
            if action_type == "command":
                success, error_msg = self._run_shell_command(action_value)
            elif action_type == "hotkey" or action_type == "preset":
                success, error_msg = self._send_hotkey(action_value)
            elif action_type == "minimize_window":
                success, error_msg = self.minimize_active_window(action_value)
            elif action_type == "restore_window":
                success, error_msg = self.restore_last_window(action_value)
            elif action_type == "resize_window_expand":
                success, error_msg = self.resize_window_expand(action_value)
            elif action_type == "resize_window_shrink":
                success, error_msg = self.resize_window_shrink(action_value)
            elif action_type == "show_desktop":
                success, error_msg = self.show_desktop(action_value)
            elif action_type == "switch_desktop_left":
                success, error_msg = self.switch_desktop_left(action_value)
            elif action_type == "switch_desktop_right":
                success, error_msg = self.switch_desktop_right(action_value)
            elif action_type == "scroll_up":
                self._execute_scroll("up", int(action_value) if action_value.isdigit() else 3)
                success = True
            elif action_type == "scroll_down":
                self._execute_scroll("down", int(action_value) if action_value.isdigit() else 3)
                success = True
            elif action_type == "zoom_in":
                self._execute_zoom("in")
                success = True
            elif action_type == "zoom_out":
                self._execute_zoom("out")
                success = True
            elif action_type == "click":
                if PYNPUT_AVAILABLE and pynput_mouse:
                    btn = Button.right if "right" in action_value.lower() else Button.left
                    pynput_mouse.click(btn)
                    success = True
                elif self.xdotool_path:
                    btn = "3" if "right" in action_value.lower() else "1"
                    subprocess.run([self.xdotool_path, "click", btn], capture_output=True, timeout=1)
                    success = True
            else:
                # Default to hotkey if formatted like a key
                success, error_msg = self._send_hotkey(action_value)

        except Exception as ex:
            error_msg = str(ex)
            success = False

        duration = round((time.time() - start_time) * 1000, 1)

        event = {
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S"),
            "gesture": gesture,
            "type": action_type,
            "value": action_value,
            "description": description,
            "success": success,
            "duration_ms": duration,
            "error": error_msg
        }

        with self._lock:
            self.history.appendleft(event)

        logger.info(f"Dispatched {gesture} -> {action_type}:{action_value} (Success={success})")

        if self.on_action_dispatched:
            try:
                self.on_action_dispatched(event)
            except Exception as e:
                logger.error(f"Error in action dispatch callback: {e}")

        return success

    def _run_shell_command(self, cmd: str) -> (bool, Optional[str]):
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=4
            )
            if res.returncode == 0:
                return True, None
            else:
                return False, res.stderr.strip() or f"Exit code {res.returncode}"
        except subprocess.TimeoutExpired:
            return False, "Command timed out after 4s"
        except Exception as e:
            return False, str(e)

    def _send_hotkey(self, combo: str) -> (bool, Optional[str]):
        combo = combo.strip()

        # Try xdotool first (most reliable on Linux X11/Xwayland)
        if self.xdotool_path:
            xdotool_key = self._format_for_xdotool(combo)
            try:
                cmd = [self.xdotool_path, "key", "--clearmodifiers", xdotool_key]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    return True, None
                logger.warning(f"xdotool returned {res.returncode}: {res.stderr}")
            except Exception as e:
                logger.warning(f"xdotool failed: {e}")

        # Fallback to pynput
        if PYNPUT_AVAILABLE and pynput_kb:
            try:
                return self._send_hotkey_pynput(combo)
            except Exception as e:
                return False, f"pynput error: {e}"

        return False, "Neither xdotool nor pynput could execute hotkey"

    def _format_for_xdotool(self, combo: str) -> str:
        parts = [p.strip() for p in combo.replace("+", " ").split()]
        mapped_parts = []
        for p in parts:
            low = p.lower()
            if low in ("ctrl", "control"):
                mapped_parts.append("ctrl")
            elif low in ("alt", "option"):
                mapped_parts.append("alt")
            elif low in ("shift",):
                mapped_parts.append("shift")
            elif low in ("super", "win", "windows", "meta"):
                mapped_parts.append("super")
            elif low in ("page_down", "pagedown", "pgdn"):
                mapped_parts.append("Page_Down")
            elif low in ("page_up", "pageup", "pgup"):
                mapped_parts.append("Page_Up")
            elif low in ("esc", "escape"):
                mapped_parts.append("Escape")
            elif low in ("enter", "return"):
                mapped_parts.append("Return")
            elif low in ("space", "spacebar"):
                mapped_parts.append("space")
            elif low in ("tab",):
                mapped_parts.append("Tab")
            elif low in ("plus", "+"):
                mapped_parts.append("plus")
            elif low in ("minus", "-"):
                mapped_parts.append("minus")
            elif low in ("equal", "="):
                mapped_parts.append("equal")
            else:
                mapped_parts.append(p)
        return "+".join(mapped_parts)

    def _send_hotkey_pynput(self, combo: str) -> (bool, Optional[str]):
        parts = [p.strip() for p in combo.replace("+", " ").split()]
        keys_to_press = []

        for p in parts:
            low = p.lower()
            if low in ("ctrl", "control"):
                keys_to_press.append(Key.ctrl)
            elif low in ("alt", "option"):
                keys_to_press.append(Key.alt)
            elif low in ("shift",):
                keys_to_press.append(Key.shift)
            elif low in ("super", "meta", "win"):
                keys_to_press.append(Key.cmd)
            elif low in ("page_down", "pagedown", "pgdn"):
                keys_to_press.append(Key.page_down)
            elif low in ("page_up", "pageup", "pgup"):
                keys_to_press.append(Key.page_up)
            elif low in ("space", "spacebar"):
                keys_to_press.append(Key.space)
            elif low in ("tab",):
                keys_to_press.append(Key.tab)
            elif low in ("enter", "return"):
                keys_to_press.append(Key.enter)
            elif low in ("esc", "escape"):
                keys_to_press.append(Key.esc)
            elif low in ("plus", "+"):
                keys_to_press.append('+')
            elif low in ("minus", "-"):
                keys_to_press.append('-')
            elif len(p) == 1:
                keys_to_press.append(p)
            else:
                if hasattr(Key, low):
                    keys_to_press.append(getattr(Key, low))
                else:
                    keys_to_press.append(p)

        for k in keys_to_press:
            pynput_kb.press(k)
        time.sleep(0.04)
        for k in reversed(keys_to_press):
            pynput_kb.release(k)

        return True, None

    def get_recent_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.history)
