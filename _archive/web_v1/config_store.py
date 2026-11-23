import copy
import json
import os
import re
import threading
from typing import Any, Dict, List

import yaml

from utils.logger import get_logger

logger = get_logger("config_store")

CAMERA_TYPES = ("video", "rtsp", "webcam")
TRACKERS = ("bytetrack", "botsort")
DEVICES = ("auto", "cpu", "cuda", "mps")
DEFAULT_COLOR = "#c8c8c8"

DEFAULT_MODEL: Dict[str, Any] = {
    "path": "",
    "device": "auto",
    "imgsz": 640,
    "conf_threshold": 0.4,
    "iou_threshold": 0.45,
    "half": False,
    "tracking": {"enabled": True, "method": "bytetrack"},
    "alert": {"min_confidence": 0.55, "min_frames": 3, "cooldown_seconds": 10},
    "classes": [],
}

# Display names used by the original pipeline (camera_worker.py)
LEGACY_ALERT_LABELS = {"ambulance": "Emergency Vehicle Detected"}


def hex_to_bgr(value: str) -> tuple:
    """'#rrggbb' -> (b, g, r) for OpenCV."""
    value = (value or DEFAULT_COLOR).lstrip("#")
    if len(value) != 6:
        value = DEFAULT_COLOR.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _bgr_to_hex(bgr) -> str:
    b, g, r = (int(c) for c in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


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


class ConfigStore:
    """
    Camera + model settings kept as two JSON files in the data folder.
    All writes are atomic (temp file + rename) and guarded by a lock.
    """

    def __init__(self, data_dir: str, legacy_dir: str = "configs"):
        os.makedirs(data_dir, exist_ok=True)
        self.cameras_file = os.path.join(data_dir, "cameras.json")
        self.model_file = os.path.join(data_dir, "model.json")
        self._lock = threading.Lock()
        self._seed_from_yaml(legacy_dir)

    # ---------------- file helpers ----------------

    @staticmethod
    def _read(path: str, default):
        if not os.path.exists(path):
            return copy.deepcopy(default)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write(path: str, data):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _seed_from_yaml(self, legacy_dir: str):
        """First run: import configs/cameras.yaml and configs/model.yaml."""
        cam_yaml = os.path.join(legacy_dir, "cameras.yaml")
        if not os.path.exists(self.cameras_file):
            cameras = []
            if os.path.exists(cam_yaml):
                with open(cam_yaml) as f:
                    raw = yaml.safe_load(f) or {}
                for c in raw.get("cameras") or []:
                    try:
                        cameras.append(self._clean_camera(c, cam_id=c.get("id")))
                    except ValueError as e:
                        logger.warning(f"Skipping camera from {cam_yaml}: {e}")
                logger.info(f"Imported {len(cameras)} camera(s) from {cam_yaml}")
            self._write(self.cameras_file, {"cameras": cameras})

        model_yaml = os.path.join(legacy_dir, "model.yaml")
        if not os.path.exists(self.model_file):
            model = copy.deepcopy(DEFAULT_MODEL)
            if os.path.exists(model_yaml):
                with open(model_yaml) as f:
                    raw = yaml.safe_load(f) or {}
                m = raw.get("model", {})
                tracking = raw.get("tracking", {})
                colors = raw.get("colors", {})
                alert_classes = raw.get("alert_classes", [])
                method = tracking.get("method", "bytetrack")
                model.update({
                    "path": m.get("path", ""),
                    "device": m.get("device", "auto"),
                    "imgsz": m.get("imgsz", 640),
                    "conf_threshold": m.get("conf_threshold", 0.4),
                    "iou_threshold": m.get("iou_threshold", 0.45),
                    "half": m.get("half", False),
                    "tracking": {
                        "enabled": tracking.get("enabled", True),
                        "method": method if method in TRACKERS else "bytetrack",
                    },
                })
                model["alert"]["min_frames"] = tracking.get("min_frames_before_alert", 3)
                model["classes"] = [
                    {
                        "name": name,
                        "color": _bgr_to_hex(colors.get(name, colors.get("default", [200, 200, 200]))),
                        "show": True,
                        "alert": name in alert_classes,
                        "alert_label": LEGACY_ALERT_LABELS.get(name, name),
                    }
                    for name in raw.get("all_classes", [])
                ]
                logger.info(f"Imported model settings from {model_yaml}")
            self._write(self.model_file, self._clean_model(model))

    # ---------------- cameras ----------------

    def _clean_camera(self, data: dict, cam_id: str) -> dict:
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
            return self._read(self.cameras_file, {"cameras": []})["cameras"]

    def get_camera(self, cam_id: str) -> dict:
        for cam in self.get_cameras():
            if cam["id"] == cam_id:
                return cam
        raise KeyError(cam_id)

    def add_camera(self, data: dict) -> dict:
        with self._lock:
            cameras = self._read(self.cameras_file, {"cameras": []})["cameras"]
            taken = {c["id"] for c in cameras}
            base = _slug(data.get("name", "cam"))
            cam_id, n = base, 2
            while cam_id in taken:
                cam_id, n = f"{base}_{n}", n + 1
            cam = self._clean_camera(data, cam_id)
            cameras.append(cam)
            self._write(self.cameras_file, {"cameras": cameras})
        logger.info(f"Camera added: {cam_id}")
        return cam

    def update_camera(self, cam_id: str, data: dict) -> dict:
        with self._lock:
            cameras = self._read(self.cameras_file, {"cameras": []})["cameras"]
            for i, c in enumerate(cameras):
                if c["id"] == cam_id:
                    cameras[i] = self._clean_camera({**c, **data}, cam_id)
                    self._write(self.cameras_file, {"cameras": cameras})
                    logger.info(f"Camera updated: {cam_id}")
                    return cameras[i]
        raise KeyError(cam_id)

    def delete_camera(self, cam_id: str):
        with self._lock:
            cameras = self._read(self.cameras_file, {"cameras": []})["cameras"]
            remaining = [c for c in cameras if c["id"] != cam_id]
            if len(remaining) == len(cameras):
                raise KeyError(cam_id)
            self._write(self.cameras_file, {"cameras": remaining})
        logger.info(f"Camera deleted: {cam_id}")

    # ---------------- model ----------------

    def _clean_model(self, data: dict) -> dict:
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
            "tracking": {
                "enabled": _as_bool(tracking.get("enabled", True)),
                "method": method,
            },
            "alert": {
                "min_confidence": _as_number(alert, "min_confidence", float, 0.01, 1.0, 0.55),
                "min_frames": _as_number(alert, "min_frames", int, 1, 1000, 3),
                "cooldown_seconds": _as_number(alert, "cooldown_seconds", float, 0, 86400, 10),
            },
            "classes": classes,
        }

    def get_model(self) -> dict:
        with self._lock:
            return self._read(self.model_file, DEFAULT_MODEL)

    def save_model(self, data: dict) -> dict:
        model = self._clean_model(data)
        with self._lock:
            self._write(self.model_file, model)
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
