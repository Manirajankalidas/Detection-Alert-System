import cv2
import math
import numpy as np
from typing import Dict, Optional
from pipeline.orchestrator import Orchestrator
from utils.logger import get_logger

logger = get_logger("dashboard")

TITLE = "Alert Monitoring System"
HEADER_H = 45
FOOTER_H = 35

BG_COLOR = (18, 18, 18)
HEADER_COLOR = (28, 28, 28)
BORDER_COLOR = (50, 50, 50)
ACCENT_COLOR = (0, 200, 140)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_text_centered(img, text, y, color, scale=0.55, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


class Dashboard:
    def __init__(
        self,
        orchestrator: Orchestrator,
        max_cols: int = 3,
        window_title: str = TITLE,
    ):
        self.orchestrator = orchestrator
        self.max_cols = max_cols
        self.window_title = window_title

        # ✅ FULLSCREEN WINDOW
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.window_title,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

        # Detect screen size dynamically
        self.screen_w = 1920
        self.screen_h = 1080

        logger.info("Dashboard initialized in FULLSCREEN")

    def _compute_layout(self, n: int):
        cols = min(n, self.max_cols)
        rows = math.ceil(n / cols)
        return rows, cols

    def _build_canvas(self, frames: Dict[str, Optional[np.ndarray]], stats) -> np.ndarray:
        cam_ids = list(frames.keys())
        n = len(cam_ids)

        if n == 0:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            _put_text_centered(canvas, "No cameras connected", 240, (200, 200, 200))
            return canvas

        rows, cols = self._compute_layout(n)

        # ✅ Dynamic tile size (fits FULL screen)
        tile_w = self.screen_w // cols
        tile_h = (self.screen_h - HEADER_H - FOOTER_H) // rows

        total_w = cols * tile_w
        total_h = rows * tile_h + HEADER_H + FOOTER_H

        canvas = np.full((total_h, total_w, 3), BG_COLOR, dtype=np.uint8)

        # ---------- HEADER ----------
        cv2.rectangle(canvas, (0, 0), (total_w, HEADER_H), HEADER_COLOR, -1)
        cv2.line(canvas, (0, HEADER_H), (total_w, HEADER_H), ACCENT_COLOR, 1)

        total_alerts = sum(s["alerts"] for s in stats)
        header_text = f"{TITLE}  |  Cameras: {n}  |  Total Alerts: {total_alerts}"
        _put_text_centered(canvas, header_text, 28, ACCENT_COLOR, scale=0.6)

        # ---------- TILES ----------
        stats_map = {s["cam_id"]: s for s in stats}

        for idx, cam_id in enumerate(cam_ids):
            row = idx // cols
            col = idx % cols

            x_off = col * tile_w
            y_off = HEADER_H + row * tile_h

            frame = frames[cam_id]
            tile = np.full((tile_h, tile_w, 3), (30, 30, 30), dtype=np.uint8)

            if frame is not None:
                resized = cv2.resize(frame, (tile_w, tile_h))
                tile = resized
            else:
                _put_text_centered(tile, "No Signal", tile_h // 2, (120, 120, 120))

            # Border
            cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), BORDER_COLOR, 1)

            # Status Dot
            s = stats_map.get(cam_id, {})
            connected = s.get("connected", False)
            dot_color = (0, 220, 100) if connected else (0, 0, 200)
            cv2.circle(tile, (tile_w - 15, 15), 6, dot_color, -1)

            # Camera Name Overlay
            cam_name = s.get("cam_name", cam_id)
            cv2.putText(tile, cam_name, (10, 25), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            canvas[y_off:y_off + tile_h, x_off:x_off + tile_w] = tile

        # ---------- FOOTER ----------
        footer_y = total_h - FOOTER_H
        cv2.rectangle(canvas, (0, footer_y), (total_w, total_h), HEADER_COLOR, -1)
        cv2.line(canvas, (0, footer_y), (total_w, footer_y), BORDER_COLOR, 1)

        for idx, cam_id in enumerate(cam_ids):
            s = stats_map.get(cam_id, {})
            txt = f"{s.get('cam_name', cam_id)} | FPS:{s.get('fps', 0)} | Alerts:{s.get('alerts', 0)}"

            x_pos = (idx * total_w // n) + 10
            cv2.putText(canvas, txt, (x_pos, footer_y + 22), FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        return canvas

    def run(self):
        logger.info("Dashboard running (FULLSCREEN). Press Q or ESC to quit.")

        while True:
            frames = self.orchestrator.get_all_frames()
            stats = self.orchestrator.get_all_stats()

            canvas = self._build_canvas(frames, stats)

            cv2.imshow(self.window_title, canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # ESC
                logger.info("Exit pressed")
                break

        cv2.destroyAllWindows()