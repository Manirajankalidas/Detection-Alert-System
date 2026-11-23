import os
import cv2
import torch
from utils.logger import get_logger

logger = get_logger("hardware")


def detect_device() -> str:
    """
    Auto-detect best available compute device.
    Returns: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {name}")
        return "cuda"

    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS (Metal) detected")
        return "mps"

    # Jetson detection via environment
    if os.path.exists("/etc/nv_tegra_release"):
        logger.info("NVIDIA Jetson (Tegra) detected - using CPU with ONNX optimization")
        return "cpu"

    logger.info("No GPU found, using CPU")
    return "cpu"


def get_optimal_workers(num_cameras: int) -> int:
    """
    Compute optimal thread workers based on CPU cores and camera count.
    Nano devices get fewer workers to avoid OOM.
    """
    cpu_count = os.cpu_count() or 2
    is_jetson = os.path.exists("/etc/nv_tegra_release")

    if is_jetson:
        # Nano has 4 cores - be conservative
        return min(num_cameras, max(1, cpu_count - 1))

    return min(num_cameras, cpu_count)


def optimize_opencv_for_device():
    """
    Apply global OpenCV optimizations based on detected hardware.
    """
    cv2.setUseOptimized(True)
    device = detect_device()

    if device == "cpu":
        # Limit OpenCV threads on nano devices to avoid starvation
        cores = os.cpu_count() or 2
        cv2.setNumThreads(max(1, cores // 2))
        logger.info(f"OpenCV threads set to {max(1, cores // 2)}")
    else:
        cv2.setNumThreads(0)  # Let OpenCV auto-manage on GPU systems

    logger.info("OpenCV optimizations applied")
