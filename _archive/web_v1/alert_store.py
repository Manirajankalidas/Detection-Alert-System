import json
import os
import queue
import re
import shutil
import threading
import uuid
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

from utils.image_utils import resize_frame
from utils.logger import get_logger

logger = get_logger("alert_store")

ALERT_STATUSES = ("new", "acknowledged")


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "x"


class AlertStore:
    """
    Alerts are kept as:
      data/alerts.jsonl                                  one JSON record per line
      data/alerts/<YYYY-MM-DD>/<camera>/<time>_<class>_<track>.jpg        full frame
      data/alerts/<YYYY-MM-DD>/<camera>/<time>_<class>_<track>_thumb.jpg  small preview
    New alerts are pushed to subscribers (the dashboard's live event stream).
    """

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

    def add(
        self,
        frame: np.ndarray,
        camera: dict,
        class_name: str,
        label: str,
        confidence: float,
        track_id: Optional[int],
        bbox: dict,
    ) -> dict:
        now = datetime.now()
        folder = os.path.join(now.strftime("%Y-%m-%d"), _safe(camera["name"]))
        os.makedirs(os.path.join(self.image_root, folder), exist_ok=True)

        stem = f"{now:%H-%M-%S}-{now.microsecond // 1000:03d}_{_safe(class_name)}_{track_id if track_id is not None else 'na'}"
        image_rel = os.path.join(folder, f"{stem}.jpg")
        thumb_rel = os.path.join(folder, f"{stem}_thumb.jpg")
        cv2.imwrite(os.path.join(self.image_root, image_rel), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(self.image_root, thumb_rel), resize_frame(frame, width=320), [cv2.IMWRITE_JPEG_QUALITY, 75])

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

    def list(
        self,
        limit: int = 100,
        camera_id: Optional[str] = None,
        class_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[dict]:
        with self._lock:
            items = list(reversed(self._alerts))
        if camera_id:
            items = [a for a in items if a["camera_id"] == camera_id]
        if class_name:
            items = [a for a in items if a["class_name"] == class_name]
        if status:
            items = [a for a in items if a["status"] == status]
        return items[:limit]

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
