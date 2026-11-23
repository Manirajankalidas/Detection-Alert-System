import base64
import json
import os
import queue

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from utils.logger import get_logger
from web.alert_store import AlertStore
from web.config_store import ConfigStore
from web.engine import DetectionEngine

logger = get_logger("web")


def _placeholder(text: str) -> bytes:
    img = np.full((360, 640, 3), (32, 28, 24), dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, text, ((640 - tw) // 2, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 140, 140), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _grab_frame(cam: dict):
    """Open a camera once and return one JPEG frame (used by the 'Test' button)."""
    source = int(cam["source"]) if cam["type"] == "webcam" else cam["source"]
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return None, "Cannot open source"
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, "Opened, but could not read a frame"
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        return cv2.imencode(".jpg", frame)[1].tobytes(), f"OK - {w}x{h} @ {fps:.0f} fps"
    finally:
        cap.release()


def create_app(config: ConfigStore, alerts: AlertStore, engine: DetectionEngine) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    placeholders = {
        "stopped": _placeholder("Detection stopped"),
        "waiting": _placeholder("Waiting for video..."),
    }

    def bad_request(e):
        return jsonify({"error": str(e)}), 400

    def not_found(what="Not found"):
        return jsonify({"error": what}), 404

    def apply_changes():
        """Config changed: restart detection if it is running."""
        if engine.running:
            engine.restart()
            return True
        return False

    # ---------------- page ----------------

    @app.get("/")
    def index():
        return render_template("index.html")

    # ---------------- cameras ----------------

    @app.get("/api/cameras")
    def list_cameras():
        return jsonify(config.get_cameras())

    @app.post("/api/cameras")
    def add_camera():
        try:
            cam = config.add_camera(request.get_json(force=True) or {})
        except ValueError as e:
            return bad_request(e)
        return jsonify({"camera": cam, "restarted": apply_changes()}), 201

    @app.put("/api/cameras/<cam_id>")
    def update_camera(cam_id):
        try:
            cam = config.update_camera(cam_id, request.get_json(force=True) or {})
        except ValueError as e:
            return bad_request(e)
        except KeyError:
            return not_found("Camera not found")
        return jsonify({"camera": cam, "restarted": apply_changes()})

    @app.delete("/api/cameras/<cam_id>")
    def delete_camera(cam_id):
        try:
            config.delete_camera(cam_id)
        except KeyError:
            return not_found("Camera not found")
        return jsonify({"deleted": cam_id, "restarted": apply_changes()})

    @app.post("/api/cameras/test")
    def test_camera():
        cam = request.get_json(force=True) or {}
        if not str(cam.get("source", "")).strip():
            return bad_request("Source is required")
        cam.setdefault("type", "video")
        if cam["type"] == "webcam" and not str(cam["source"]).isdigit():
            return bad_request("Webcam source must be a device index like 0")
        jpeg, message = _grab_frame(cam)
        if jpeg is None:
            return jsonify({"ok": False, "message": message})
        return jsonify({"ok": True, "message": message,
                        "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()})

    @app.get("/api/cameras/<cam_id>/frame.jpg")
    def camera_frame(cam_id):
        jpeg = engine.latest_jpeg(cam_id)
        if jpeg is None:
            jpeg = placeholders["waiting" if engine.running else "stopped"]
        return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    # ---------------- model ----------------

    @app.get("/api/model")
    def get_model():
        model = config.get_model()
        return jsonify({"model": model, "warnings": config.model_warnings(model)})

    @app.put("/api/model")
    def save_model():
        try:
            model = config.save_model(request.get_json(force=True) or {})
        except ValueError as e:
            return bad_request(e)
        return jsonify({"model": model, "warnings": config.model_warnings(model),
                        "restarted": apply_changes()})

    @app.post("/api/model/classes")
    def read_model_classes():
        """Read class names straight from a .pt file."""
        path = str((request.get_json(force=True) or {}).get("path", "")).strip()
        if not os.path.isfile(path):
            return bad_request(f"Model file not found: {path or '(empty)'}")
        try:
            from ultralytics import YOLO
            names = YOLO(path).names
        except Exception as e:
            return bad_request(f"Could not read model: {e}")
        return jsonify({"classes": [names[i] for i in sorted(names)]})

    # ---------------- engine ----------------

    @app.get("/api/engine/status")
    def engine_status():
        return jsonify(engine.status())

    @app.post("/api/engine/start")
    def engine_start():
        engine.start()
        return jsonify(engine.status())

    @app.post("/api/engine/stop")
    def engine_stop():
        engine.stop()
        return jsonify(engine.status())

    # ---------------- alerts ----------------

    @app.get("/api/alerts")
    def list_alerts():
        try:
            limit = min(int(request.args.get("limit", 100)), 1000)
        except ValueError:
            return bad_request("limit must be a number")
        return jsonify({
            "alerts": alerts.list(
                limit=limit,
                camera_id=request.args.get("camera") or None,
                class_name=request.args.get("class") or None,
                status=request.args.get("status") or None,
            ),
            "counts": alerts.counts(),
        })

    @app.patch("/api/alerts/<alert_id>")
    def update_alert(alert_id):
        try:
            alert = alerts.set_status(alert_id, (request.get_json(force=True) or {}).get("status", ""))
        except ValueError as e:
            return bad_request(e)
        except KeyError:
            return not_found("Alert not found")
        return jsonify(alert)

    @app.post("/api/alerts/acknowledge-all")
    def acknowledge_all():
        return jsonify({"acknowledged": alerts.acknowledge_all()})

    @app.delete("/api/alerts/<alert_id>")
    def delete_alert(alert_id):
        try:
            alerts.delete(alert_id)
        except KeyError:
            return not_found("Alert not found")
        return jsonify({"deleted": alert_id})

    @app.delete("/api/alerts")
    def clear_alerts():
        alerts.clear()
        return jsonify({"cleared": True})

    @app.get("/api/alerts/events")
    def alert_events():
        """Server-Sent Events: pushes every new alert to the dashboard."""
        q = alerts.subscribe()

        def stream():
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        record = q.get(timeout=15)
                        yield f"event: alert\ndata: {json.dumps(record)}\n\n"
                    except queue.Empty:
                        yield ": ping\n\n"   # keeps the connection alive
            finally:
                alerts.unsubscribe(q)

        return Response(stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/media/alerts/<path:rel>")
    def alert_image(rel):
        path = alerts.file_path(rel)
        if not path or not os.path.isfile(path):
            abort(404)
        return send_file(path, mimetype="image/jpeg", max_age=86400)

    return app
