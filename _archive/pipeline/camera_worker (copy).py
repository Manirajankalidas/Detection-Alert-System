import time
import threading
import numpy as np
from typing import Dict, Optional, List

from api.config import CameraConfig, ModelConfig, MongoConfig, ClientConfig
from api.database import get_collection
from core.detector import YOLODetector, Detection
from core.tracker import TrackMemory, get_tracker_yaml
from core.alert_manager import AlertManager
from core.stream_manager import StreamManager
from utils.image_utils import draw_box
from utils.logger import get_logger

logger = get_logger("camera_worker")

# BGR color mapping
DEFAULT_COLORS = {
    "car":       (144, 238, 144),
    "truck":     (173, 216, 230),
    "bike":      (255, 255, 153),
    "bus":       (255, 182, 193),
    "others":    (221, 160, 221),
    "ambulance": (0, 80, 255),
}
DEFAULT_COLOR = (200, 200, 200)


class CameraWorker:

    def __init__(
        self,
        cam_cfg: CameraConfig,
        model_cfg: ModelConfig,
        mongo_cfg: MongoConfig,
        client_cfg: ClientConfig,
        detector: YOLODetector,
    ):
        self.cam_cfg = cam_cfg
        self.model_cfg = model_cfg

        self.stream = StreamManager(
            source=cam_cfg.source,
            cam_id=cam_cfg.id,
            stream_type=cam_cfg.type,
        )

        self.detector = detector

        self.track_memory = TrackMemory(
            min_frames=model_cfg.min_frames_before_alert
        )

        self.alert_manager = AlertManager(
            client_name=client_cfg.name,
            reports_collection=mongo_cfg.reports_collection,
            clients_collection=mongo_cfg.clients_collection,
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._fps: float = 0.0
        self._alert_count: int = 0
        self._status: str = "idle"

    def start(self):
        self.stream.start()
        self._running = True
        self._thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name=f"worker-{self.cam_cfg.id}"
        )
        self._thread.start()
        logger.info(f"[{self.cam_cfg.id}] Worker started")

    def stop(self):
        self._running = False
        self.stream.stop()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info(f"[{self.cam_cfg.id}] Worker stopped")

    def _process_loop(self):
        frame_count = 0
        fps_timer = time.time()
        use_tracking = self.model_cfg.tracking_enabled
        tracker_yaml = get_tracker_yaml(self.model_cfg.tracking_method)

        while self._running:
            ret, frame = self.stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self._status = "running"
            annotated = frame.copy()

            detections: List[Detection] = self.detector.detect(
                frame,
                use_tracking=use_tracking,
                tracker=tracker_yaml,
            )

            active_ids = set()

            for det in detections:
                cls = det.class_name

                # 🔥 ONLY CHANGE (SAFE MAPPING)
                if cls == "ambulance":
                    violation_name = "Emergency Vehicle Detected"
                else:
                    violation_name = cls

                color = self.model_cfg.colors.get(cls, DEFAULT_COLOR)
                if isinstance(color, list):
                    color = tuple(color)

                label = f"{violation_name}"
                if det.track_id is not None:
                    label += f" #{det.track_id}"
                label += f" {det.confidence:.0%}"

                draw_box(
                    annotated,
                    det.x1, det.y1, det.x2, det.y2,
                    label=label,
                    color=color,
                )

                if cls not in self.model_cfg.alert_classes:
                    continue

                track_id = det.track_id if det.track_id is not None else -1

                if use_tracking:
                    active_ids.add(track_id)
                    should_alert = self.track_memory.update(track_id)
                else:
                    should_alert = True

                if should_alert:
                    self.alert_manager.raise_alert(
                        frame=frame,
                        cam_name=self.cam_cfg.name,
                        location=self.cam_cfg.location,
                        class_name=violation_name,  # ✅ mapped name
                        track_id=det.track_id,
                        confidence=det.confidence,
                        x1=det.x1, y1=det.y1,
                        x2=det.x2, y2=det.y2,
                        save_crop=True,
                    )
                    self._alert_count += 1

            if use_tracking:
                self.track_memory.reset_lost(active_ids)

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                self._fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            self._draw_overlay(annotated)

            with self._frame_lock:
                self._latest_frame = annotated

    def _draw_overlay(self, frame: np.ndarray):
        import cv2
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        status_text = (
            f"{self.cam_cfg.name} | "
            f"{self.cam_cfg.location} | "
            f"FPS:{self._fps:.1f} | "
            f"Alerts:{self._alert_count}"
        )
        cv2.putText(
            frame, status_text,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 180),
            1,
            cv2.LINE_AA,
        )

    def get_frame(self):
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def stats(self) -> dict:
        return {
            "cam_id": self.cam_cfg.id,
            "cam_name": self.cam_cfg.name,
            "fps": round(self._fps, 1),
            "alerts": self._alert_count,
            "status": self._status,
            "connected": self.stream.is_connected,
        }