"""
Detection engine - one thread pair per enabled camera:

  VideoStream (reader thread)  ->  CameraRunner (detection thread)
     reads frames at source FPS       YOLO + tracking -> draw boxes -> alert rules -> AlertStore
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from storage import AlertStore, ConfigStore, hex_to_bgr

logger = logging.getLogger("engine")

PREVIEW_WIDTH = 960          # live preview is downscaled to save bandwidth
ALERT_BOX_COLOR = (0, 0, 255)
UNKNOWN_CLASS = {"color": "#c8c8c8", "show": True, "alert": False}
TRACKER_FILES = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}


# ======================= drawing =======================

def draw_box(frame, x1, y1, x2, y2, label, color, thickness=1, font_scale=0.5):
    """Bounding box with a filled label. Label goes inside the box if it would leave the frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    weight = max(1, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, weight)
    ly = y1 if y1 - th - 6 >= 0 else y1 + th + 6
    cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 4, ly), color, -1)
    cv2.putText(frame, label, (x1 + 2, ly - 3), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), weight, cv2.LINE_AA)


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(h * width / w)))


# ======================= video input =======================

class VideoStream:
    """
    Reads a video file / RTSP stream / webcam in its own thread at the source FPS.
    - keeps the last few frames in a small buffer (old frames are dropped if detection is slower)
    - video files loop forever
    - RTSP/webcam reconnect automatically
    """

    def __init__(self, source, cam_id: str, stream_type: str = "video", reconnect_delay: float = 3.0):
        self.source = source
        self.cam_id = cam_id
        self.stream_type = stream_type
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._buffer = deque(maxlen=5)
        self.fps: float = 25.0
        self.is_connected = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"[{self.cam_id}] Stream started | source={self.source}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self._buffer:
                return False, None
            return True, self._buffer.popleft()

    def _open(self) -> bool:
        if self._cap:
            self._cap.release()
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.warning(f"[{self.cam_id}] Cannot open source")
            return False
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 25.0
        self.is_connected = True
        logger.info(f"[{self.cam_id}] Connected | FPS={self.fps:.1f}")
        return True

    def _read_loop(self):
        while self._running:
            if not self.is_connected:
                if not self._open():
                    time.sleep(self.reconnect_delay)
                    continue

            start = time.time()
            ret, frame = self._cap.read()
            if not ret:
                if self.stream_type == "video":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop the file
                    continue
                logger.warning(f"[{self.cam_id}] Frame read failed - reconnecting")
                self.is_connected = False
                continue

            with self._lock:
                self._buffer.append(frame)

            # play at real-time speed
            sleep = 1.0 / self.fps - (time.time() - start)
            if sleep > 0:
                time.sleep(sleep)


# ======================= YOLO =======================

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    """YOLO wrapper. Not thread-safe - each camera gets its own instance."""

    def __init__(self, model_cfg: dict):
        device = pick_device(model_cfg["device"])
        self.half = model_cfg["half"] and device == "cuda"
        self.conf = model_cfg["conf_threshold"]
        self.iou = model_cfg["iou_threshold"]
        self.imgsz = model_cfg["imgsz"]
        self.tracking = model_cfg["tracking"]["enabled"]
        self.tracker = TRACKER_FILES.get(model_cfg["tracking"]["method"], "bytetrack.yaml")

        logger.info(f"Loading YOLO model {model_cfg['path']} on {device} | half={self.half}")
        self.model = YOLO(model_cfg["path"])
        self.model.to(device)
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> List[Detection]:
        args = dict(source=frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, half=self.half, verbose=False)
        if self.tracking:
            r = self.model.track(persist=True, tracker=self.tracker, **args)[0]
        else:
            r = self.model.predict(**args)[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id
        return [
            Detection(*map(int, boxes[i]), class_name=self.names[int(classes[i])],
                      confidence=float(confs[i]), track_id=int(ids[i]) if ids is not None else None)
            for i in range(len(boxes))
        ]


# ======================= per-camera loop =======================

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
        self.stream = VideoStream(source=source, cam_id=camera["id"], stream_type=camera["type"])

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._jpeg_lock = threading.Lock()
        self._jpeg: Optional[bytes] = None          # downscaled preview (camera grid)
        self._frame: Optional[np.ndarray] = None    # full-size annotated frame (big view)

        self.status = "starting"
        self.error = ""
        self.fps = 0.0
        self.alert_count = 0

        # alert confirmation state
        self._streak: Dict[tuple, int] = {}       # key -> consecutive frames seen
        self._alerted: set = set()                # keys that already raised an alert
        self._last_alert: Dict[str, float] = {}   # class -> last alert time (cooldown)

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

    def latest_jpeg(self, full: bool = False) -> Optional[bytes]:
        with self._jpeg_lock:
            if not full:
                return self._jpeg
            frame = self._frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

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

    def _run(self):
        self.status = "loading model"
        try:
            detector = Detector(self.model)
        except Exception as e:
            self.status = "error"
            self.error = f"Model load failed: {e}"
            logger.error(f"[{self.camera['id']}] {self.error}")
            return

        if not self._running:   # stopped while the model was loading
            return

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
                detections = detector.detect(frame)
                self.status, self.error = "running", ""
            except Exception as e:
                self.status = "error"
                self.error = f"Detection failed: {e}"
                logger.error(f"[{self.camera['id']}] {self.error}")
                time.sleep(1.0)
                continue

            annotated = frame.copy()
            self._process(frame, annotated, detections, detector.tracking)

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                self.fps = frame_count / (now - fps_timer)
                frame_count, fps_timer = 0, now

            preview = resize_to_width(annotated, PREVIEW_WIDTH) if annotated.shape[1] > PREVIEW_WIDTH else annotated
            ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._jpeg_lock:
                    self._jpeg = buf.tobytes()
                    self._frame = annotated

        self.stream.stop()   # covers a stop() that raced with stream.start()

    def _process(self, frame: np.ndarray, annotated: np.ndarray, detections: List[Detection], use_tracking: bool):
        rules = self.model["alert"]
        seen = set()
        font = max(0.5, frame.shape[1] / 1400)   # readable labels on large frames too
        line = max(1, frame.shape[1] // 900)

        for det in detections:
            cls = self.classes.get(det.class_name, UNKNOWN_CLASS)
            if not cls["show"]:
                continue

            is_alert_class = cls["alert"]
            label = cls.get("alert_label", det.class_name) if is_alert_class else det.class_name
            text = label + (f" #{det.track_id}" if det.track_id is not None else "") + f" {det.confidence:.0%}"
            draw_box(annotated, det.x1, det.y1, det.x2, det.y2, label=text, color=hex_to_bgr(cls["color"]),
                     thickness=line * 3 if is_alert_class else line, font_scale=font)

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

    def _raise(self, frame: np.ndarray, det: Detection, label: str):
        snapshot = frame.copy()
        w = frame.shape[1]
        draw_box(snapshot, det.x1, det.y1, det.x2, det.y2, label=f"{label} {det.confidence:.0%}",
                 color=ALERT_BOX_COLOR, thickness=max(3, w // 400), font_scale=max(0.7, w / 1100))
        bbox = {"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2}
        try:
            self.alerts.add(snapshot, self.camera, det.class_name, label, det.confidence, det.track_id, bbox)
            self.alert_count += 1
        except Exception as e:
            logger.error(f"[{self.camera['id']}] Saving alert failed: {e}")


# ======================= all cameras =======================

class DetectionEngine:
    """Starts/stops one CameraRunner per enabled camera using the saved settings."""

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

    def latest_jpeg(self, cam_id: str, full: bool = False) -> Optional[bytes]:
        with self._lock:
            r = self._runners.get(cam_id)
        return r.latest_jpeg(full) if r else None
