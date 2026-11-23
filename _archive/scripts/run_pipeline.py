#!/usr/bin/env python3
"""
Entry Point: Start all camera workers + REST API + optional GUI.
Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --no-gui
"""

import argparse
import threading
import signal
import sys
import os

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from api.config import load_config
from api.database import init_db, close_db
from api.routes import api_bp
from pipeline.orchestrator import Orchestrator
from utils.hardware import optimize_opencv_for_device
from utils.logger import get_logger

logger = get_logger("main")


def create_app(orchestrator: Orchestrator) -> Flask:
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    return app


def main():
    parser = argparse.ArgumentParser(description="Alert Monitoring System")
    parser.add_argument("--no-gui", action="store_true", help="Run headless (no GUI)")
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--cameras-config", default="configs/cameras.yaml")
    args = parser.parse_args()

    logger.info("=== Alert Monitoring System Starting ===")

    # Load configs
    config = load_config(args.config, args.model_config, args.cameras_config)
    logger.info(f"Config loaded | Cameras: {len(config.cameras)}")

    # Hardware optimizations
    optimize_opencv_for_device()

    # Init MongoDB
    init_db(config.mongo.uri, config.mongo.database)

    # Start orchestrator
    orchestrator = Orchestrator(config)
    orchestrator.start()

    # Start Flask API in background thread
    flask_app = create_app(orchestrator)
    api_thread = threading.Thread(
        target=lambda: flask_app.run(
            host=config.api_host,
            port=config.api_port,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
        name="flask-api"
    )
    api_thread.start()
    logger.info(f"REST API running on http://{config.api_host}:{config.api_port}")

    # Graceful shutdown
    def shutdown(sig, frame):
        logger.info("Shutdown signal received...")
        orchestrator.stop()
        close_db()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start GUI (blocks main thread)
    if not args.no_gui and config.gui_enabled:

        from gui.dashboard import Dashboard

        dashboard = Dashboard(
            orchestrator=orchestrator,
            max_cols=config.gui_max_cols,
        )

        dashboard.run()
    else:
        logger.info("Running headless. Ctrl+C to stop.")
        signal.pause()

    orchestrator.stop()
    close_db()
    logger.info("=== System Shutdown Complete ===")


if __name__ == "__main__":
    main()
