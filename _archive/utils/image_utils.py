import cv2
import base64
import numpy as np
from typing import Optional, Tuple


def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """
    Encode an OpenCV BGR frame to a data URI base64 JPEG string.
    Quality: 1-100, lower = smaller file.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def resize_frame(
    frame: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    inter: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resize frame preserving aspect ratio if only one dimension is given."""
    h, w = frame.shape[:2]
    if width is None and height is None:
        return frame
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    elif height is None:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    else:
        dim = (width, height)
    return cv2.resize(frame, dim, interpolation=inter)


def draw_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 1,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding box with label background on frame. Returns modified frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    # Label sits above the box; if there is no room (box at the top edge) put it inside
    ly = y1 if y1 - th - 6 >= 0 else y1 + th + 6
    cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 4, ly), color, -1)
    cv2.putText(
        frame, label,
        (x1 + 2, ly - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )
    return frame


def crop_detection(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """Safe crop with boundary clamping."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2].copy()
