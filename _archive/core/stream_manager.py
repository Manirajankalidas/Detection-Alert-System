import cv2
import time
import threading
import numpy as np
from typing import Optional, Tuple
from collections import deque
from utils.logger import get_logger

logger = get_logger("stream_manager")


class StreamManager:
    """
    Smooth video stream manager:
    - FPS synchronized
    - Buffered frames (no jitter)
    - Video looping
    - Works for RTSP / file / webcam
    """

    def __init__(
        self,
        source,
        cam_id: str,
        stream_type: str = "video",
        reconnect_delay: float = 3.0,
    ):
        self.source = source
        self.cam_id = cam_id
        self.stream_type = stream_type
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._fps: float = 25.0
        self._connected = False

        # 🔥 FRAME BUFFER (KEY FOR SMOOTHNESS)
        self._buffer = deque(maxlen=5)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"[{self.cam_id}] Stream started | source={self.source}")

    def _open_capture(self) -> bool:
        if self._cap:
            self._cap.release()

        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.warning(f"[{self.cam_id}] Cannot open source")
            return False

        self._cap = cap

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 25.0

        self._fps = fps
        self._connected = True

        logger.info(f"[{self.cam_id}] Connected | FPS={self._fps:.1f}")
        return True

    def _read_loop(self):
        while self._running:

            if not self._connected:
                if not self._open_capture():
                    time.sleep(self.reconnect_delay)
                    continue

            frame_delay = 1.0 / self._fps

            start_time = time.time()

            ret, frame = self._cap.read()

            if not ret:
                # 🔥 LOOP VIDEO SMOOTHLY
                if self.stream_type == "video":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                logger.warning(f"[{self.cam_id}] Frame read failed")
                self._connected = False
                continue

            with self._lock:
                self._buffer.append(frame)

            # 🔥 FPS SYNC (REAL-TIME PLAYBACK)
            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self._buffer:
                return False, None
            return True, self._buffer.popleft()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        logger.info(f"[{self.cam_id}] Stream stopped")

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_connected(self) -> bool:
        return self._connected