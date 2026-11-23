from utils.logger import get_logger

logger = get_logger("tracker")


class TrackMemory:
    """
    Lightweight in-memory tracker for consecutive-frame confirmation.
    Works with or without ByteTrack/DeepSORT.
    Avoids duplicate alerts for the same track ID.
    """

    def __init__(self, min_frames: int = 3):
        self.min_frames = min_frames
        self._frame_counts: dict = {}
        self._alerted_ids: set = set()

    def update(self, track_id: int) -> bool:
        """
        Increment frame count for track_id.
        Returns True when threshold is first reached (trigger alert once).
        """
        self._frame_counts[track_id] = self._frame_counts.get(track_id, 0) + 1

        if (
            self._frame_counts[track_id] >= self.min_frames
            and track_id not in self._alerted_ids
        ):
            self._alerted_ids.add(track_id)
            logger.debug(f"Track {track_id} confirmed after {self.min_frames} frames")
            return True
        return False

    def reset_lost(self, active_ids: set):
        """Zero out counters for IDs no longer visible."""
        lost = set(self._frame_counts.keys()) - active_ids
        for lid in lost:
            self._frame_counts[lid] = 0

    def clear(self):
        self._frame_counts.clear()
        self._alerted_ids.clear()


def get_tracker_yaml(method: str) -> str:
    """
    Return the tracker config yaml file name for YOLO tracking.
    """
    mapping = {
        "bytetrack": "bytetrack.yaml",
        "deepsort": "deepocsort.yaml",
        "botsort": "botsort.yaml",
    }
    return mapping.get(method.lower(), "bytetrack.yaml")
