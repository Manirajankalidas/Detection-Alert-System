"""
Everything that is saved to disk (no database):

  data/cameras.json      camera list               (Cameras page)
  data/model.json        model + alert settings    (Model page)
  data/alerts.jsonl      one alert per line
  data/alerts/<YYYY-MM-DD>/<camera>/<time>_<class>_<track>.jpg (+ _thumb.jpg)
"""

import copy
import json
import logging
import os
import queue
import re
import shutil
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("storage")

CAMERA_TYPES = ("video", "rtsp", "webcam")
TRACKERS = ("bytetrack", "botsort")
DEVICES = ("auto", "cpu", "cuda", "mps")
ALERT_STATUSES = ("new", "acknowledged")
DEFAULT_COLOR = "#c8c8c8"


def _cls(name, color, alert=False, label=None):
    return {"name": name, "color": color, "show": True, "alert": alert, "alert_label": label or name}


# First-run settings: the bundled car_fall model with "ambulance" as the alert class
DEFAULT_MODEL: Dict[str, Any] = {
    "path": "models/car_fall.pt",
    "device": "auto",
    "imgsz": 640,
    "conf_threshold": 0.4,
    "iou_threshold": 0.45,
    "half": False,
    "tracking": {"enabled": True, "method": "bytetrack"},
    "alert": {"min_confidence": 0.55, "min_frames": 3, "cooldown_seconds": 10},
    "classes": [
        _cls("car", "#90ee90"),
        _cls("truck", "#e6d8ad"),
        _cls("bike", "#99ffff"),
        _cls("bus", "#c1b6ff"),
        _cls("others", "#dda0dd"),
        _cls("ambulance", "#ff5000", alert=True, label="Emergency Vehicle Detected"),
        _cls("car_fall", "#ffa500"),
    ],
}


# ---------------- small helpers ----------------

def hex_to_bgr(value: str) -> tuple:
    """'#rrggbb' -> (b, g, r) for OpenCV."""
    value = (value or DEFAULT_COLOR).lstrip("#")
    if len(value) != 6:
        value = DEFAULT_COLOR.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "x"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "cam"


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _as_number(data: dict, key: str, cast, lo, hi, default):
    raw = data.get(key, default)
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number")
    if not lo <= value <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}")
    return value


def _read_json(path: str, default):
    if not os.path.exists(path):
        return copy.deepcopy(default)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data):
    tmp = path + ".tmp"          # write + rename, so a crash never leaves half a file
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ======================= settings =======================

class ConfigStore:
    """Camera + model settings, validated and saved as JSON."""

    def __init__(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.cameras_file = os.path.join(data_dir, "cameras.json")
        self.model_file = os.path.join(data_dir, "model.json")
        self._lock = threading.Lock()
        if not os.path.exists(self.cameras_file):
            _write_json(self.cameras_file, {"cameras": []})
        if not os.path.exists(self.model_file):
            _write_json(self.model_file, DEFAULT_MODEL)

    # ---------------- cameras ----------------

    @staticmethod
    def _clean_camera(data: dict, cam_id: str) -> dict:
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError("Camera name is required")
        cam_type = str(data.get("type", "video")).strip().lower()
        if cam_type not in CAMERA_TYPES:
            raise ValueError(f"Camera type must be one of {', '.join(CAMERA_TYPES)}")
        source = str(data.get("source", "")).strip()
        if source == "":
            raise ValueError("Camera source is required")
        if cam_type == "webcam" and not source.isdigit():
            raise ValueError("Webcam source must be a device index like 0 or 1")
        return {
            "id": cam_id,
            "name": name,
            "location": str(data.get("location", "")).strip() or name,
            "source": source,
            "type": cam_type,
            "enabled": _as_bool(data.get("enabled", True)),
        }

    def get_cameras(self) -> List[dict]:
        with self._lock:
            return _read_json(self.cameras_file, {"cameras": []})["cameras"]

    def add_camera(self, data: dict) -> dict:
        with self._lock:
            cameras = _read_json(self.cameras_file, {"cameras": []})["cameras"]
            taken = {c["id"] for c in cameras}
            base = _slug(data.get("name", "cam"))
            cam_id, n = base, 2
            while cam_id in taken:
                cam_id, n = f"{base}_{n}", n + 1
            cam = self._clean_camera(data, cam_id)
            cameras.append(cam)
            _write_json(self.cameras_file, {"cameras": cameras})
        logger.info(f"Camera added: {cam_id}")
        return cam

    def update_camera(self, cam_id: str, data: dict) -> dict:
        with self._lock:
            cameras = _read_json(self.cameras_file, {"cameras": []})["cameras"]
            for i, c in enumerate(cameras):
                if c["id"] == cam_id:
                    cameras[i] = self._clean_camera({**c, **data}, cam_id)
                    _write_json(self.cameras_file, {"cameras": cameras})
                    logger.info(f"Camera updated: {cam_id}")
                    return cameras[i]
        raise KeyError(cam_id)

    def delete_camera(self, cam_id: str):
        with self._lock:
            cameras = _read_json(self.cameras_file, {"cameras": []})["cameras"]
            remaining = [c for c in cameras if c["id"] != cam_id]
            if len(remaining) == len(cameras):
                raise KeyError(cam_id)
            _write_json(self.cameras_file, {"cameras": remaining})
        logger.info(f"Camera deleted: {cam_id}")

    # ---------------- model ----------------

    @staticmethod
    def _clean_model(data: dict) -> dict:
        tracking = data.get("tracking") or {}
        alert = data.get("alert") or {}
        device = str(data.get("device", "auto")).lower()
        if device not in DEVICES:
            raise ValueError(f"Device must be one of {', '.join(DEVICES)}")
        method = str(tracking.get("method", "bytetrack")).lower()
        if method not in TRACKERS:
            raise ValueError(f"Tracker must be one of {', '.join(TRACKERS)}")

        imgsz = _as_number(data, "imgsz", int, 32, 4096, 640)
        imgsz = max(32, round(imgsz / 32) * 32)   # YOLO needs a multiple of 32

        classes, seen = [], set()
        for c in data.get("classes") or []:
            name = str(c.get("name", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            color = str(c.get("color", DEFAULT_COLOR))
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                color = DEFAULT_COLOR
            classes.append({
                "name": name,
                "color": color.lower(),
                "show": _as_bool(c.get("show", True)),
                "alert": _as_bool(c.get("alert", False)),
                "alert_label": str(c.get("alert_label", "")).strip() or name,
            })

        return {
            "path": str(data.get("path", "")).strip(),
            "device": device,
            "imgsz": imgsz,
            "conf_threshold": _as_number(data, "conf_threshold", float, 0.01, 1.0, 0.4),
            "iou_threshold": _as_number(data, "iou_threshold", float, 0.01, 1.0, 0.45),
            "half": _as_bool(data.get("half", False)),
            "tracking": {"enabled": _as_bool(tracking.get("enabled", True)), "method": method},
            "alert": {
                "min_confidence": _as_number(alert, "min_confidence", float, 0.01, 1.0, 0.55),
                "min_frames": _as_number(alert, "min_frames", int, 1, 1000, 3),
                "cooldown_seconds": _as_number(alert, "cooldown_seconds", float, 0, 86400, 10),
            },
            "classes": classes,
        }

    def get_model(self) -> dict:
        with self._lock:
            return _read_json(self.model_file, DEFAULT_MODEL)

    def save_model(self, data: dict) -> dict:
        model = self._clean_model(data)
        with self._lock:
            _write_json(self.model_file, model)
        logger.info("Model settings saved")
        return model

    @staticmethod
    def model_warnings(model: dict) -> List[str]:
        warnings = []
        if not model["path"]:
            warnings.append("No model file set.")
        elif not os.path.isfile(model["path"]):
            warnings.append(f"Model file not found: {model['path']}")
        if not any(c["alert"] for c in model["classes"]):
            warnings.append("No class is marked as an alert class - no alerts will be raised.")
        return warnings


# ======================= alerts =======================

class AlertStore:
    """Saves alert images + records, and pushes new alerts to open dashboards."""

    def __init__(self, data_dir: str):
        self.image_root = os.path.join(data_dir, "alerts")
        self.index_file = os.path.join(data_dir, "alerts.jsonl")
        os.makedirs(self.image_root, exist_ok=True)
        self._lock = threading.Lock()
        self._subscribers: List[queue.Queue] = []
        self._alerts: List[dict] = self._load()
        logger.info(f"Loaded {len(self._alerts)} alert(s) from {self.index_file}")

    def _load(self) -> List[dict]:
        alerts = []
        if not os.path.exists(self.index_file):
            return alerts
        with open(self.index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    alerts.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupt line in alerts.jsonl")
        return alerts

    def _rewrite(self):
        tmp = self.index_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for a in self._alerts:
                f.write(json.dumps(a) + "\n")
        os.replace(tmp, self.index_file)

    # ---------------- write ----------------

    def add(self, frame: np.ndarray, camera: dict, class_name: str, label: str,
            confidence: float, track_id: Optional[int], bbox: dict) -> dict:
        now = datetime.now()
        folder = os.path.join(now.strftime("%Y-%m-%d"), _safe(camera["name"]))
        os.makedirs(os.path.join(self.image_root, folder), exist_ok=True)

        stem = f"{now:%H-%M-%S}-{now.microsecond // 1000:03d}_{_safe(class_name)}_{track_id if track_id is not None else 'na'}"
        image_rel = os.path.join(folder, f"{stem}.jpg")
        thumb_rel = os.path.join(folder, f"{stem}_thumb.jpg")
        h, w = frame.shape[:2]
        thumb = cv2.resize(frame, (320, max(1, int(h * 320 / w))))
        cv2.imwrite(os.path.join(self.image_root, image_rel), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(self.image_root, thumb_rel), thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])

        record = {
            "id": uuid.uuid4().hex[:12],
            "time": now.isoformat(timespec="seconds"),
            "camera_id": camera["id"],
            "camera_name": camera["name"],
            "location": camera["location"],
            "class_name": class_name,
            "label": label,
            "confidence": round(float(confidence), 4),
            "track_id": track_id,
            "bbox": bbox,
            "image": image_rel.replace(os.sep, "/"),
            "thumb": thumb_rel.replace(os.sep, "/"),
            "status": "new",
        }

        with self._lock:
            self._alerts.append(record)
            with open(self.index_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            subscribers = list(self._subscribers)

        for q in subscribers:
            try:
                q.put_nowait(record)
            except queue.Full:
                pass  # slow/dead browser tab - never block the camera thread
        logger.info(f"ALERT {label} | {camera['name']} | conf={confidence:.2f} | track={track_id}")
        return record

    def set_status(self, alert_id: str, status: str) -> dict:
        if status not in ALERT_STATUSES:
            raise ValueError(f"Status must be one of {', '.join(ALERT_STATUSES)}")
        with self._lock:
            for a in self._alerts:
                if a["id"] == alert_id:
                    a["status"] = status
                    self._rewrite()
                    return a
        raise KeyError(alert_id)

    def acknowledge_all(self) -> int:
        with self._lock:
            changed = 0
            for a in self._alerts:
                if a["status"] == "new":
                    a["status"] = "acknowledged"
                    changed += 1
            if changed:
                self._rewrite()
        return changed

    def delete(self, alert_id: str):
        with self._lock:
            for i, a in enumerate(self._alerts):
                if a["id"] == alert_id:
                    del self._alerts[i]
                    self._rewrite()
                    break
            else:
                raise KeyError(alert_id)
        for rel in (a.get("image"), a.get("thumb")):
            path = self.file_path(rel) if rel else None
            if path and os.path.isfile(path):
                os.remove(path)

    def clear(self):
        """Delete every alert record and image."""
        with self._lock:
            self._alerts = []
            self._rewrite()
            shutil.rmtree(self.image_root, ignore_errors=True)
            os.makedirs(self.image_root, exist_ok=True)
        logger.info("All alerts cleared")

    # ---------------- read ----------------

    def query(self, camera_id: Optional[str] = None, class_name: Optional[str] = None,
              status: Optional[str] = None, date_from: Optional[str] = None,
              date_to: Optional[str] = None) -> List[dict]:
        """All matching alerts, newest first. Dates are 'YYYY-MM-DD' (inclusive)."""
        with self._lock:
            items = list(reversed(self._alerts))
        if camera_id:
            items = [a for a in items if a["camera_id"] == camera_id]
        if class_name:
            items = [a for a in items if a["class_name"] == class_name]
        if status:
            items = [a for a in items if a["status"] == status]
        if date_from:
            items = [a for a in items if a["time"][:10] >= date_from]
        if date_to:
            items = [a for a in items if a["time"][:10] <= date_to]
        return items

    def violation_types(self) -> List[dict]:
        """Distinct alert classes seen so far (for the filter dropdown)."""
        with self._lock:
            types = {a["class_name"]: a["label"] for a in self._alerts}
        return [{"name": n, "label": types[n]} for n in sorted(types)]

    def counts(self) -> dict:
        with self._lock:
            total = len(self._alerts)
            new = sum(1 for a in self._alerts if a["status"] == "new")
        return {"total": total, "new": new}

    def file_path(self, rel: str) -> Optional[str]:
        """Resolve an image path inside the alerts folder (blocks ../ escapes)."""
        root = os.path.realpath(self.image_root)
        path = os.path.realpath(os.path.join(root, rel))
        return path if path.startswith(root + os.sep) else None

    # ---------------- live push ----------------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)
