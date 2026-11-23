import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional
from utils.logger import get_logger
from utils.hardware import detect_device

logger = get_logger("detector")


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None


class YOLODetector:
    """
    Thread-safe YOLO inference wrapper.
    Single model instance shared via load once — do NOT call from
    multiple threads simultaneously. Each camera worker has its own.
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.55,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        half: bool = False,
    ):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        resolved_device = detect_device() if device == "auto" else device
        self.half = half and (resolved_device == "cuda")

        logger.info(f"Loading YOLO model on {resolved_device} | half={self.half}")
        self.model = YOLO(model_path)
        self.model.to(resolved_device)
        self.names = self.model.names
        logger.info(f"Model loaded. Classes: {list(self.names.values())}")

    def detect(self, frame: np.ndarray, use_tracking: bool = False, tracker: str = "bytetrack.yaml") -> List[Detection]:
        """
        Run inference on a single frame.
        Returns list of Detection objects.
        Skips inference if frame is empty.
        """
        if frame is None or frame.size == 0:
            return []

        # old 

        # infer_kwargs = dict(
        #     source=frame,
        #     conf=self.conf,
        #     iou=self.iou,
        #     imgsz=self.imgsz,
        #     half=self.half,
        #     verbose=False,
        # )

        # New - sep11 Mani
        infer_kwargs = dict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
        )

        if use_tracking:
            results = self.model.track(
                persist=True,
                tracker=tracker,
                **infer_kwargs
            )
        else:
            results = self.model.predict(**infer_kwargs)

        return self._parse_results(results)

    def _parse_results(self, results) -> List[Detection]:
        detections = []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            track_id = int(ids[i]) if ids is not None else None
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                class_id=int(classes[i]),
                class_name=self.names[int(classes[i])],
                confidence=float(confs[i]),
                track_id=track_id,
            ))

        return detections
