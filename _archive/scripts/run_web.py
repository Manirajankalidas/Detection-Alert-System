#!/usr/bin/env python3
"""
Entry Point: web-based detection + alert dashboard (no database, JSON storage).
Usage:
    python scripts/run_web.py
    python scripts/run_web.py --port 8000 --no-autostart
Then open http://localhost:8000
"""

import argparse
import os
import sys

# Run from the project root so relative paths (videos, model, data/, logs/) always resolve the same way
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from utils.hardware import optimize_opencv_for_device
from utils.logger import get_logger
from web.alert_store import AlertStore
from web.config_store import ConfigStore
from web.engine import DetectionEngine
from web.server import create_app

logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser(description="Web Detection Alert System")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="data", help="Where configs, alerts and images are stored")
    parser.add_argument("--no-autostart", action="store_true", help="Don't start detection on launch")
    args = parser.parse_args()

    logger.info("=== Web Detection Alert System Starting ===")
    optimize_opencv_for_device()

    config = ConfigStore(args.data_dir)
    alerts = AlertStore(args.data_dir)
    engine = DetectionEngine(config, alerts)

    if not args.no_autostart and config.get_cameras():
        engine.start()

    app = create_app(config, alerts, engine)
    logger.info(f"Dashboard: http://localhost:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    finally:
        engine.stop()
        logger.info("=== Shutdown Complete ===")


if __name__ == "__main__":
    main()
