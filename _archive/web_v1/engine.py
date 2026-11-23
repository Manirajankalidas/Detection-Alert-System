import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.detector import YOLODetector
from core.stream_manager import StreamManager
from core.tracker import get_tracker_yaml
from utils.image_utils import draw_box, resize_frame
from utils.logger import get_logger
from web.alert_store import AlertStore
from web.config_store import ConfigStore, hex_to_bgr

logger = get_logger("engine")

PREVIEW_WIDTH = 960          # live preview is downscaled to save bandwidth
ALERT_BOX_COLOR = (0, 0, 255)
UNKNOWN_CLASS = {"color": "#c8c8c8", "show": True, "alert": False}


class CameraRunner:
    """
    One thread per camera:
      read frame -> YOLO (+tracking) -> draw boxes -> confirm alert -> save alert
    The latest annotated frame is kept as JPEG bytes for the dashboard.
    """

    def __init__(self, camera: dict, model: dict, alerts: AlertStore):
        self.camera = camera
        self.model = model
        self.alerts = alerts
        self.classes = {c["name"]: c for c in model["classes"]}

        source = int(camera["source"]) if camera["type"] == "webcam" else camera["source"]
        self.stream = StreamManager(source=source, cam_id=camera["id"], stream_type=camera["type"])

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._jpeg_lock = threading.Lock()
        self._jpeg: Optional[bytes] = None

        self.status = "starting"
        self.error = ""
        self.fps = 0.0
        self.alert_count = 0

        # alert confirmation state
        self._streak: Dict[tuple, int] = {}       # key -> consecutive frames seen
        self._alerted: set = set()                # keys that already raised an alert
        self._last_alert: Dict[str, float] = {}   # class -> last alert time (cooldown)

    # ---------------- lifecycle ----------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"runner-{self.camera['id']}")
        self._thread.start()

    def stop(self):
        self._running = False
        self.stream.stop()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.status = "stopped"

    def latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._jpeg

    @property
    def stats(self) -> dict:
        return {
            "id": self.camera["id"],
            "name": self.camera["name"],
            "location": self.camera["location"],
            "status": self.status,
            "error": self.error,
            "connected": self.stream.is_connected,
            "fps": round(self.fps, 1),
            "alerts": self.alert_count,
        }

    # ---------------- main loop ----------------

    def _run(self):
        m = self.model
        self.status = "loading model"
        try:
            detector = YOLODetector(
                model_path=m["path"],
                conf=m["conf_threshold"],
                iou=m["iou_threshold"],
                imgsz=m["imgsz"],
                device=m["device"],
                half=m["half"],
            )
        except Exception as e:
            self.status = "error"
            self.error = f"Model load failed: {e}"
            logger.error(f"[{self.camera['id']}] {self.error}")
            return

        if not self._running:   # stopped while the model was loading
            return

        use_tracking = m["tracking"]["enabled"]
        tracker_yaml = get_tracker_yaml(m["tracking"]["method"])

        self.stream.start()
        self.status = "connecting"
        frame_count, fps_timer = 0, time.time()

        while self._running:
            ret, frame = self.stream.read()
            if not ret or frame is None:
                if not self.stream.is_connected:
                    self.status = "connecting"
                time.sleep(0.01)
                continue

            try:
                detections = detector.detect(frame, use_tracking=use_tracking, tracker=tracker_yaml)
                self.status, self.error = "running", ""
            except Exception as e:
                self.status = "error"
                self.error = f"Detection failed: {e}"
                logger.error(f"[{self.camera['id']}] {self.error}")
                time.sleep(1.0)
                continue

            annotated = frame.copy()
            self._process(frame, annotated, detections, use_tracking)

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                self.fps = frame_count / (now - fps_timer)
                frame_count, fps_timer = 0, now

            preview = resize_frame(annotated, width=PREVIEW_WIDTH) if annotated.shape[1] > PREVIEW_WIDTH else annotated
            ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._jpeg_lock:
                    self._jpeg = buf.tobytes()

        self.stream.stop()   # covers a stop() that raced with stream.start()

    def _process(self, frame: np.ndarray, annotated: np.ndarray, detections, use_tracking: bool):
        rules = self.model["alert"]
        seen = set()

        for det in detections:
            cls = self.classes.get(det.class_name, UNKNOWN_CLASS)
            if not cls["show"]:
                continue

            is_alert_class = cls["alert"]
            label = cls.get("alert_label", det.class_name) if is_alert_class else det.class_name
            text = label + (f" #{det.track_id}" if det.track_id is not None else "") + f" {det.confidence:.0%}"
            draw_box(annotated, det.x1, det.y1, det.x2, det.y2, label=text,
                     color=hex_to_bgr(cls["color"]), thickness=3 if is_alert_class else 1)

            if not is_alert_class or det.confidence < rules["min_confidence"]:
                continue

            # With tracking each vehicle is counted separately; without it we count per class.
            tracked = use_tracking and det.track_id is not None
            key = (det.class_name, det.track_id if tracked else None)
            seen.add(key)
            self._streak[key] = self._streak.get(key, 0) + 1

            if self._streak[key] < rules["min_frames"] or key in self._alerted:
                continue
            now = time.time()
            if now - self._last_alert.get(det.class_name, 0) < rules["cooldown_seconds"]:
                continue

            self._raise(frame, det, label)
            self._last_alert[det.class_name] = now
            self._alerted.add(key)

        # Consecutive-frame streaks break when the object disappears.
        for key in list(self._streak):
            if key not in seen:
                del self._streak[key]
                if key[1] is None:
                    self._alerted.discard(key)   # untracked: allow a new alert next time it appears
        if len(self._alerted) > 10000:
            self._alerted.clear()

    def _raise(self, frame: np.ndarray, det, label: str):
        snapshot = frame.copy()
        draw_box(snapshot, det.x1, det.y1, det.x2, det.y2,
                 label=f"{label} {det.confidence:.0%}", color=ALERT_BOX_COLOR, thickness=3, font_scale=0.7)
        bbox = {"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2}
        try:
            self.alerts.add(snapshot, self.camera, det.class_name, label, det.confidence, det.track_id, bbox)
            self.alert_count += 1
        except Exception as e:
            logger.error(f"[{self.camera['id']}] Saving alert failed: {e}")


class DetectionEngine:
    """Starts/stops one CameraRunner per enabled camera using the saved config."""

    def __init__(self, config: ConfigStore, alerts: AlertStore):
        self.config = config
        self.alerts = alerts
        self._runners: Dict[str, CameraRunner] = {}
        self._lock = threading.Lock()
        self.running = False

    def start(self):
        with self._lock:
            if self.running:
                return
            model = self.config.get_model()
            cameras = [c for c in self.config.get_cameras() if c["enabled"]]
            for cam in cameras:
                runner = CameraRunner(cam, model, self.alerts)
                runner.start()
                self._runners[cam["id"]] = runner
            self.running = True
        logger.info(f"Detection started on {len(cameras)} camera(s)")

    def stop(self):
        with self._lock:
            runners = list(self._runners.values())
            self._runners = {}
            self.running = False
        for r in runners:
            r.stop()
        logger.info("Detection stopped")

    def restart(self):
        self.stop()
        self.start()

    def status(self) -> dict:
        with self._lock:
            runners = dict(self._runners)
        cameras: List[dict] = []
        for cam in self.config.get_cameras():
            r = runners.get(cam["id"])
            if r:
                cameras.append(r.stats)
            else:
                cameras.append({
                    "id": cam["id"], "name": cam["name"], "location": cam["location"],
                    "status": "disabled" if not cam["enabled"] else "stopped",
                    "error": "", "connected": False, "fps": 0.0, "alerts": 0,
                })
        return {"running": self.running, "cameras": cameras, "alerts": self.alerts.counts()}

    def latest_jpeg(self, cam_id: str) -> Optional[bytes]:
        with self._lock:
            r = self._runners.get(cam_id)
        return r.latest_jpeg() if r else None
