import time
from typing import List, Dict, Optional
from api.config import AppConfig, CameraConfig
from core.detector import YOLODetector
from pipeline.camera_worker import CameraWorker
from utils.logger import get_logger
from utils.hardware import get_optimal_workers

logger = get_logger("orchestrator")


class Orchestrator:
    """
    Manages lifecycle of all CameraWorker instances.
    Each camera gets its OWN YOLODetector instance to avoid thread contention.
    Provides unified stats and frame access for the GUI.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.workers: Dict[str, CameraWorker] = {}
        self._running = False

    def start(self):
        logger.info(f"Starting orchestrator for {len(self.config.cameras)} cameras")

        max_workers = get_optimal_workers(len(self.config.cameras))
        cameras_to_start = self.config.cameras[:max_workers]

        if len(self.config.cameras) > max_workers:
            logger.warning(
                f"Hardware limit: starting {max_workers} of "
                f"{len(self.config.cameras)} cameras"
            )

        for cam_cfg in cameras_to_start:
            self._start_worker(cam_cfg)

        self._running = True
        logger.info("Orchestrator started all workers")

    def _start_worker(self, cam_cfg: CameraConfig):
        """Create a fresh detector per camera (thread isolation)."""
        detector = YOLODetector(
            model_path=self.config.model.path,
            conf=self.config.model.conf_threshold,
            iou=self.config.model.iou_threshold,
            imgsz=self.config.model.imgsz,
            device=self.config.model.device,
            half=self.config.model.half,
        )

        worker = CameraWorker(
            cam_cfg=cam_cfg,
            model_cfg=self.config.model,
            mongo_cfg=self.config.mongo,
            client_cfg=self.config.client,
            detector=detector,
        )
        worker.start()
        self.workers[cam_cfg.id] = worker
        logger.info(f"Worker started for camera: {cam_cfg.id}")

    def stop(self):
        logger.info("Stopping all workers...")
        for worker in self.workers.values():
            worker.stop()
        self.workers.clear()
        self._running = False
        logger.info("Orchestrator stopped")

    def get_all_frames(self) -> Dict[str, any]:
        """Returns {cam_id: frame} for all active workers."""
        return {
            cam_id: worker.get_frame()
            for cam_id, worker in self.workers.items()
        }

    def get_all_stats(self) -> List[dict]:
        return [w.stats for w in self.workers.values()]
