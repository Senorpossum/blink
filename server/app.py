"""
FastAPI Server & WebSocket Streaming Engine for BlinkFlow & HandFlow.
Coordinates camera worker, hand gesture detector, eye blink detector, action dispatcher, and client dashboard.
"""

import asyncio
import base64
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from blink_core import (
    ActionConfig,
    ActionDispatcher,
    AppConfig,
    BlinkDetector,
    CameraWorker,
    ConfigManager,
    DetectionResult,
    FaceGesture,
    GestureEngine,
    HandAction,
    HandData,
    HandDetectionResult,
    HandDetector,
    HandGestureEngine,
)

logger = logging.getLogger("VisionServer")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="HandFlow & BlinkFlow API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
class SystemState:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load()
        self.dispatcher = ActionDispatcher()

        # Hand detection & gesture engine
        self.hand_detector = HandDetector(model_path="gesture_recognizer.task")
        h_sens = self.config.hand_sensitivity
        self.hand_gesture_engine = HandGestureEngine(
            scroll_deadzone=h_sens.scroll_deadzone,
            scroll_inverted=h_sens.scroll_inverted,
            swipe_velocity_thresh=h_sens.swipe_velocity_thresh,
            pose_hold_ms=h_sens.pose_hold_ms,
            input_delay_ms=h_sens.input_delay_ms,
            air_mouse_enabled=getattr(h_sens, "air_mouse_enabled", True),
            air_mouse_idle_sec=getattr(h_sens, "air_mouse_idle_sec", 2.0),
            dwell_click_ms=getattr(h_sens, "dwell_click_ms", 700),
            screen_width=self.dispatcher.screen_width,
            screen_height=self.dispatcher.screen_height,
        )

        # Eye detection & gesture engine (for toggle)
        self.detector = BlinkDetector(model_path="face_landmarker.task")
        self.gesture_engine = GestureEngine(self.config.sensitivity)

        self.camera = CameraWorker(
            camera_index=self.config.camera_index,
            width=self.config.camera_width,
            height=self.config.camera_height,
            fps=self.config.camera_fps
        )

        self.active_sockets: Set[WebSocket] = set()
        self.latest_telemetry: Dict[str, Any] = {}
        self.latest_encoded_frame: Optional[str] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False

        # Calibration state (eyes)
        self.calibrating = False
        self.calib_phase: Optional[str] = None
        self.calib_samples_open: List[float] = []
        self.calib_samples_closed: List[float] = []
        self.last_mouth_pos: Optional[Tuple[float, float]] = None
        self.last_face_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_face_check_time: float = 0.0

        # Wire up callbacks
        self.gesture_engine.on_gesture_detected = self._handle_eye_gesture
        self.hand_gesture_engine.on_continuous_action = self._handle_hand_continuous
        self.hand_gesture_engine.on_discrete_gesture = self._handle_hand_discrete
        self.hand_gesture_engine.on_cursor_move = self._handle_cursor_move
        self.hand_gesture_engine.on_cursor_click = self._handle_cursor_click

    def _handle_cursor_move(self, sx: int, sy: int):
        if not self.config.active or self.config.mode != "hands":
            return
        self.dispatcher.move_cursor(sx, sy)

    def _handle_cursor_click(self, button: str = "left"):
        if not self.config.active or self.config.mode != "hands":
            return
        # If left click, check if AIR_CLICK is configured to a custom action (e.g. open terminal)
        if "left" in button.lower():
            mapping = self.config.hand_mappings.get("AIR_CLICK")
            if mapping and mapping.enabled and mapping.type != "click":
                self.dispatcher.dispatch(
                    gesture="AIR_CLICK",
                    action_type=mapping.type,
                    action_value=mapping.value,
                    description=mapping.description or "Air Click: Open Terminal"
                )
                return
        self.dispatcher.mouse_click(button)

    def _handle_eye_gesture(self, gesture: FaceGesture, duration_ms: float):
        if not self.config.active or self.config.mode != "eyes":
            return

        gesture_key = gesture.value
        mapping = self.config.mappings.get(gesture_key)
        if mapping and mapping.enabled and mapping.value:
            self.dispatcher.dispatch(
                gesture=gesture_key,
                action_type=mapping.type,
                action_value=mapping.value,
                description=mapping.description or mapping.value
            )

    def _handle_hand_continuous(self, action: HandAction, intensity: float):
        if not self.config.active or self.config.mode != "hands":
            return

        action_key = action.value
        mapping = self.config.hand_mappings.get(action_key)
        if mapping and mapping.enabled:
            # For continuous scroll/zoom, execute directly
            if mapping.type in ("scroll_up", "scroll_down"):
                direction = "up" if mapping.type == "scroll_up" else "down"
                steps = max(1, int(intensity))
                self.dispatcher.scroll(direction, steps)
            elif mapping.type in ("zoom_in", "zoom_out"):
                direction = "in" if mapping.type == "zoom_in" else "out"
                self.dispatcher.zoom(direction)
            elif mapping.value:
                self.dispatcher.dispatch(
                    gesture=action_key,
                    action_type=mapping.type,
                    action_value=mapping.value,
                    description=mapping.description or action_key
                )

    def _handle_hand_discrete(self, action: HandAction, meta: str):
        if not self.config.active or self.config.mode != "hands":
            return

        action_key = action.value
        mapping = self.config.hand_mappings.get(action_key)
        if mapping and mapping.enabled:
            self.dispatcher.dispatch(
                gesture=action_key,
                action_type=mapping.type,
                action_value=mapping.value,
                description=f"{mapping.description or action_key} ({meta})"
            )

state = SystemState()


# Background Vision Loop
def vision_processing_loop():
    logger.info("Starting background vision loop...")
    state.camera.start()

    while state.is_running:
        frame = state.camera.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        try:
            if state.config.mode == "hands":
                # Hand Tracking Mode
                click_feedback = state.hand_gesture_engine.last_click_event if (time.time() - state.hand_gesture_engine.last_click_time < 0.25) else None
                extra_info = {
                    "air_mouse_active": state.hand_gesture_engine.air_mouse_active,
                    "idle_progress": state.hand_gesture_engine.idle_progress,
                    "dwell_progress": state.hand_gesture_engine.dwell_progress,
                    "cursor_pos": state.hand_gesture_engine.cursor_screen_pos,
                    "engine_state": state.hand_gesture_engine.current_state_text,
                    "click_feedback": click_feedback,
                }
                res, annotated_frame = state.hand_detector.process_frame(
                    frame,
                    draw_overlay=state.config.draw_mesh,
                    extra_info=extra_info
                )

                # Check face mouth position and face bounding box for anonymity blurring
                mouth_pos = state.last_mouth_pos
                primary_pose = res.primary_hand.gesture_name if res.primary_hand else "None"
                now_t = time.time()
                need_face_check = state.config.blur_face or primary_pose == "Pointing_Up" or (now_t - state.last_face_check_time) > 0.45
                if need_face_check:
                    state.last_face_check_time = now_t
                    try:
                        face_res, _ = state.detector.process_frame(frame, draw_overlay=False)
                        if face_res.face_detected:
                            if face_res.mouth_center != (0.0, 0.0):
                                state.last_mouth_pos = face_res.mouth_center
                                mouth_pos = face_res.mouth_center
                            if face_res.face_bbox:
                                state.last_face_bbox = face_res.face_bbox
                    except Exception as e:
                        logger.debug(f"Face check error: {e}")

                # Apply anonymity face blur if enabled
                if state.config.blur_face and state.last_face_bbox:
                    state.detector.apply_face_blur(annotated_frame, state.last_face_bbox)

                # Process gesture engine with all hands and mouth position
                hand_action = state.hand_gesture_engine.process(
                    res.primary_hand,
                    all_hands=res.hands,
                    face_mouth_pos=mouth_pos
                )

                telemetry = {
                    "type": "telemetry",
                    "mode": "hands",
                    "timestamp": time.time(),
                    "active": state.config.active,
                    "blur_face": state.config.blur_face,
                    "hand_detected": res.hands_detected,
                    "num_hands": res.num_hands,
                    "air_mouse_active": state.hand_gesture_engine.air_mouse_active,
                    "idle_progress": round(state.hand_gesture_engine.idle_progress, 2),
                    "dwell_progress": round(state.hand_gesture_engine.dwell_progress, 2),
                    "cursor_pos": list(state.hand_gesture_engine.cursor_screen_pos),
                    "pose": res.primary_hand.gesture_name if res.primary_hand else "None",
                    "latency_ms": res.latency_ms,
                    "fps": state.camera.actual_fps,
                    "engine_state": state.hand_gesture_engine.current_state_text,
                    "last_gesture": hand_action.value if hand_action else None,
                    "recent_actions": state.dispatcher.get_recent_history()[:5]
                }

            else:
                # Eye Tracking Mode
                res, annotated_frame = state.detector.process_frame(
                    frame,
                    draw_overlay=state.config.draw_mesh
                )

                # Apply anonymity face blur if enabled
                if state.config.blur_face and res.face_detected and res.face_bbox:
                    state.detector.apply_face_blur(annotated_frame, res.face_bbox)

                if state.calibrating and res.face_detected:
                    avg_score = (res.left_blink_score + res.right_blink_score) / 2.0
                    if state.calib_phase == "open":
                        state.calib_samples_open.append(avg_score)
                    elif state.calib_phase == "closed":
                        state.calib_samples_closed.append(avg_score)

                gesture = state.gesture_engine.process(
                    res.left_blink_score,
                    res.right_blink_score,
                    res.face_detected
                )

                telemetry = {
                    "type": "telemetry",
                    "mode": "eyes",
                    "timestamp": time.time(),
                    "active": state.config.active,
                    "blur_face": state.config.blur_face,
                    "face_detected": res.face_detected,
                    "left_blink": res.left_blink_score,
                    "right_blink": res.right_blink_score,
                    "left_ear": res.left_ear,
                    "right_ear": res.right_ear,
                    "latency_ms": res.latency_ms,
                    "fps": state.camera.actual_fps,
                    "engine_state": state.gesture_engine.current_state_text,
                    "last_gesture": gesture.value if gesture else None,
                    "recent_actions": state.dispatcher.get_recent_history()[:5]
                }

            # Encode frame to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            state.latest_encoded_frame = jpg_as_text
            state.latest_telemetry = telemetry

        except Exception as e:
            logger.error(f"Error in vision loop: {e}")
            time.sleep(0.02)

    state.camera.stop()
    logger.info("Vision processing loop ended.")


@app.on_event("startup")
def startup_event():
    state.is_running = True
    state.worker_thread = threading.Thread(target=vision_processing_loop, daemon=True)
    state.worker_thread.start()


@app.on_event("shutdown")
def shutdown_event():
    state.is_running = False
    state.camera.stop()
    state.detector.close()
    state.hand_detector.close()


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.active_sockets.add(websocket)
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.03)
                msg = json.loads(data)
                if msg.get("action") == "toggle_active":
                    state.config.active = not state.config.active
                    state.config_manager.save()
                elif msg.get("action") == "toggle_blur_face":
                    state.config.blur_face = not state.config.blur_face
                    state.config_manager.save()
                elif msg.get("action") == "set_mode":
                    state.config.mode = msg.get("mode", "hands")
                    state.config_manager.save()
            except asyncio.TimeoutError:
                pass

            if state.latest_encoded_frame:
                payload = dict(state.latest_telemetry)
                payload["frame"] = state.latest_encoded_frame
                await websocket.send_text(json.dumps(payload))

            await asyncio.sleep(0.025)  # ~35-40 FPS max push
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        state.active_sockets.discard(websocket)


# REST Endpoints
@app.get("/api/status")
def get_status():
    return {
        "mode": state.config.mode,
        "active": state.config.active,
        "camera_connected": state.camera.connected,
        "camera_index": state.camera.camera_index,
        "fps": state.camera.actual_fps,
        "telemetry": state.latest_telemetry,
        "recent_actions": state.dispatcher.get_recent_history()
    }


@app.get("/api/config")
def get_config():
    return state.config_manager.config


class ConfigUpdateRequest(BaseModel):
    mode: Optional[str] = None
    active: Optional[bool] = None
    camera_index: Optional[int] = None
    audio_feedback: Optional[bool] = None
    draw_mesh: Optional[bool] = None
    blur_face: Optional[bool] = None
    sensitivity: Optional[Dict[str, Any]] = None
    hand_sensitivity: Optional[Dict[str, Any]] = None
    mappings: Optional[Dict[str, Any]] = None
    hand_mappings: Optional[Dict[str, Any]] = None


@app.post("/api/config")
def update_config(req: ConfigUpdateRequest):
    updates = req.model_dump(exclude_unset=True)
    updated = state.config_manager.update_from_dict(updates)
    state.config = updated
    state.gesture_engine.update_sensitivity(updated.sensitivity)
    state.hand_gesture_engine.update_sensitivity(updated.hand_sensitivity)

    if "camera_index" in updates:
        state.camera.change_camera(updated.camera_index)

    return {"status": "ok", "config": updated}


class BlurFaceRequest(BaseModel):
    blur_face: Optional[bool] = None


@app.post("/api/blur-face")
def set_blur_face(req: Optional[BlurFaceRequest] = None):
    if req and req.blur_face is not None:
        state.config.blur_face = bool(req.blur_face)
    else:
        state.config.blur_face = not state.config.blur_face
    state.config_manager.save()
    return {"status": "ok", "blur_face": state.config.blur_face}


class ModeRequest(BaseModel):
    mode: str  # "hands" or "eyes"


@app.post("/api/mode")
def set_mode(req: ModeRequest):
    if req.mode not in ("hands", "eyes"):
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'hands' or 'eyes'.")
    state.config.mode = req.mode
    state.config_manager.save()
    return {"status": "ok", "mode": state.config.mode}


@app.post("/api/toggle-active")
def toggle_active():
    state.config.active = not state.config.active
    state.config_manager.save()
    return {"status": "ok", "active": state.config.active}


class TestActionRequest(BaseModel):
    gesture: str
    type: str
    value: str
    description: Optional[str] = ""


@app.post("/api/test-action")
def test_action(req: TestActionRequest):
    success = state.dispatcher.dispatch(
        gesture=req.gesture,
        action_type=req.type,
        action_value=req.value,
        description=req.description or "Manual UI Test"
    )
    return {"status": "ok" if success else "failed", "dispatched": success}


class PresetRequest(BaseModel):
    preset: str


@app.post("/api/preset")
def apply_preset(req: PresetRequest):
    applied = state.config_manager.apply_preset(req.preset)
    if not applied:
        raise HTTPException(status_code=400, detail=f"Preset {req.preset} not found")
    state.config = state.config_manager.config
    mappings = state.config.hand_mappings if state.config.mode == "hands" else state.config.mappings
    return {"status": "ok", "preset": req.preset, "mappings": mappings}


@app.get("/api/cameras")
def list_cameras():
    cameras = CameraWorker.list_available_cameras()
    return {"available": cameras, "current": state.camera.camera_index}


class CalibrateStepRequest(BaseModel):
    phase: str  # "start_open", "start_closed", "calculate"


@app.post("/api/calibrate")
def calibrate_step(req: CalibrateStepRequest):
    if req.phase == "start_open":
        state.calib_samples_open.clear()
        state.calib_phase = "open"
        state.calibrating = True
        return {"status": "recording_open"}

    elif req.phase == "start_closed":
        state.calib_samples_closed.clear()
        state.calib_phase = "closed"
        state.calibrating = True
        return {"status": "recording_closed"}

    elif req.phase == "calculate":
        state.calibrating = False
        state.calib_phase = None

        if len(state.calib_samples_open) < 5 or len(state.calib_samples_closed) < 5:
            return {
                "status": "insufficient_data",
                "message": "Not enough sample frames captured. Please ensure your face is visible."
            }

        avg_open = sum(state.calib_samples_open) / len(state.calib_samples_open)
        avg_closed = sum(state.calib_samples_closed) / len(state.calib_samples_closed)

        recommended_thresh = round(avg_open + (avg_closed - avg_open) * 0.65, 2)
        recommended_thresh = max(0.40, min(0.85, recommended_thresh))

        recommended_wink_open = round(avg_open + (avg_closed - avg_open) * 0.30, 2)
        recommended_wink_open = max(0.20, min(0.50, recommended_wink_open))

        state.config.sensitivity.blink_threshold = recommended_thresh
        state.config.sensitivity.wink_open_threshold = recommended_wink_open
        state.config_manager.save()
        state.gesture_engine.update_sensitivity(state.config.sensitivity)

        return {
            "status": "success",
            "avg_open": round(avg_open, 3),
            "avg_closed": round(avg_closed, 3),
            "recommended_threshold": recommended_thresh,
            "recommended_wink_open": recommended_wink_open
        }

    return {"status": "unknown_phase"}


# Mount static assets
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "HandFlow backend running. Frontend static files not found."})
