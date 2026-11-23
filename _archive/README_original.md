# Alert Monitoring System (Vehicle / Ambulance Detection)

A real-time video analytics app. It watches one or more cameras (RTSP, video file or webcam), runs a **YOLOv8** model on every frame, tracks vehicles across frames, and when an **alert class** (by default `ambulance`) is confirmed it:

1. saves an annotated snapshot to `violations/<date>/<camera>/`,
2. stores a report (with a base64 image) in **MongoDB**,
3. sends an **email** with the snapshot attached.

It also shows a full-screen multi-camera **live dashboard** and serves a small **REST API** for reading and updating the saved reports.

There are now **two ways to run it**:

| | Web app (new) | Original pipeline |
|---|---|---|
| Start | `python scripts/run_web.py` | `python scripts/run_pipeline.py` |
| UI | Browser - dashboard + camera & model settings pages | OpenCV full-screen window |
| Config | Edited in the browser, saved to `data/*.json` | Hand-edited `configs/*.yaml` + `.env` |
| Storage | JSON files + images in `data/` (no database) | MongoDB + `violations/` images + email |

---

## 0. Web app (no database)

```bash
pip install -r requirements.txt
python scripts/run_web.py                 # then open http://localhost:8000
python scripts/run_web.py --port 9000 --no-autostart --data-dir data
```

| Option | Default | Meaning |
|---|---|---|
| `--host` | `0.0.0.0` | listen address (others on the network can open `http://<your-ip>:8000`) |
| `--port` | `8000` | web port |
| `--data-dir` | `data` | where settings, alerts and images are kept |
| `--no-autostart` | off | don't start detection automatically on launch |

### Pages

1. **Cameras** - add / edit / delete / enable cameras: name, location, type (`video` file, `rtsp` stream, `webcam` index) and source path. **Test connection** grabs one frame so you can check the path before saving.
2. **Model** - model `.pt` path, device, image size, FP16, detection confidence, NMS IoU, tracking on/off + tracker, and the **alert rules**:
   - *Alert confidence threshold* - a detection must be at least this confident to count towards an alert
   - *Confirm after N consecutive frames* - stops one-frame false positives
   - *Cooldown per class* - minimum seconds between two alerts of the same class on one camera
   - **Classes table** - click **Load classes from model file** to read class names from the `.pt`, then set each class's box **colour**, **Show** (draw it or hide it), **Alert** (raise alerts for it) and **Alert label** (text shown on the alert, e.g. `ambulance` → `Emergency Vehicle Detected`).
3. **Dashboard** - live annotated video for every enabled camera (FPS, alert count, connection dot), plus the **alert feed** on the right. New alerts appear instantly, with a toast, a beep (toggle *Sound*) and a red flash on the camera tile. Click a thumbnail to see the full image; ✓ acknowledges, ✕ deletes.

Saving cameras or model settings **restarts detection automatically** if it is running. Start/Stop is in the top-right corner.

### Web app flow

```
Browser ──(REST)──► web/server.py ──► ConfigStore  ── data/cameras.json, data/model.json
   ▲  ▲                      │
   │  │ frame.jpg (polled)   └──► DetectionEngine ── one CameraRunner thread per enabled camera:
   │  │                               StreamManager → YOLODetector(+ByteTrack) → draw boxes
   │  │                               → alert rules (confidence, N frames, once per track, cooldown)
   │  └── latest annotated JPEG ◄─────┘           │
   │                                              ▼
   └──(Server-Sent Events: new alert)◄── AlertStore ── data/alerts.jsonl + data/alerts/<date>/<camera>/*.jpg
```

### What gets stored (all inside `data/`)

```
data/
├── cameras.json        # camera list (edited from the Cameras page)
├── model.json          # model + alert settings (edited from the Model page)
├── alerts.jsonl        # one alert per line (JSON)
└── alerts/2026-11-01/Gate_02/
    ├── 14-03-22-415_ambulance_7.jpg         # full frame with the alert box
    └── 14-03-22-415_ambulance_7_thumb.jpg   # small preview for the dashboard
```

Example line from `alerts.jsonl`:
```json
{"id": "8e92d34727eb", "time": "2026-11-01T14:03:22", "camera_id": "gate_02", "camera_name": "Gate_02",
 "location": "Main gate", "class_name": "ambulance", "label": "Emergency Vehicle Detected",
 "confidence": 0.91, "track_id": 7, "bbox": {"x1": 151, "y1": 60, "x2": 900, "y2": 700},
 "image": "2026-11-01/Gate_02/14-03-22-415_ambulance_7.jpg",
 "thumb": "2026-11-01/Gate_02/14-03-22-415_ambulance_7_thumb.jpg", "status": "new"}
```

On the very first start, `data/cameras.json` and `data/model.json` are created from the existing `configs/cameras.yaml` and `configs/model.yaml`. After that, the YAML files are not used by the web app. To start over, stop the app and delete the `data/` folder.

### Web app files

```
scripts/run_web.py        # entry point
web/
├── server.py             # Flask routes: pages, REST API, frame.jpg, alert events, alert images
├── engine.py             # DetectionEngine + CameraRunner (detection loop and alert rules)
├── config_store.py       # cameras.json / model.json read, validate, save; first-run YAML import
├── alert_store.py        # alerts.jsonl + image files, live push to the dashboard
├── templates/index.html  # the single page
└── static/app.js, style.css
```
It reuses `core/stream_manager.py`, `core/detector.py`, `core/tracker.py` and `utils/` from the original app. It does **not** need MongoDB, so `api/` and `core/alert_manager.py` aren't used.

### Web API (for scripts)

| Method | Endpoint | |
|---|---|---|
| GET / POST | `/api/cameras` | list / add camera |
| PUT / DELETE | `/api/cameras/<id>` | edit / delete camera |
| POST | `/api/cameras/test` | try opening a source, returns one frame |
| GET | `/api/cameras/<id>/frame.jpg` | latest annotated frame |
| GET / PUT | `/api/model` | read / save model + alert settings |
| POST | `/api/model/classes` | `{"path": "..."}` → class names inside a `.pt` |
| GET | `/api/engine/status` | running flag, per-camera status/FPS/alerts, alert counts |
| POST | `/api/engine/start`, `/api/engine/stop` | control detection |
| GET | `/api/alerts?limit=&camera=&class=&status=` | alerts, newest first |
| PATCH | `/api/alerts/<id>` | `{"status": "acknowledged"}` |
| POST | `/api/alerts/acknowledge-all` | |
| DELETE | `/api/alerts/<id>`, `/api/alerts` | delete one / all (records + images) |
| GET | `/api/alerts/events` | Server-Sent Events stream of new alerts |

### Web app notes
- This uses Flask's built-in server, which is fine for a lab or a single machine. There is **no login**, so don't expose it to the internet.
- On CPU expect roughly 5-8 FPS per 1080p camera with YOLOv8n. Use a GPU (`device: cuda`) for more cameras.
- Each camera loads its own copy of the model (tracking state is per camera).

---

## 1. The big picture

```
                         ┌─────────────────────── scripts/run_pipeline.py ───────────────────────┐
                         │                                                                        │
  configs/*.yaml + .env ─┼─► load_config() ─► init_db() ─► Orchestrator.start() ─┬─► Flask API thread (port 5050)
                         │                                                       └─► Dashboard (main thread) or headless
                         └────────────────────────────────────────────────────────────────────────┘

  Orchestrator  ── creates one CameraWorker per camera (each with its OWN YOLODetector)

  CameraWorker (1 thread per camera)
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │ StreamManager ──frame──► YOLODetector ──detections──► draw boxes ──► is class an alert?     │
  │ (own read thread,        (YOLO predict / track,                          │ yes             │
  │  5-frame buffer,          ByteTrack IDs)                                  ▼                 │
  │  auto-reconnect,                                          TrackMemory: seen ≥ N frames      │
  │  loops video files)                                       and not alerted before?           │
  │                                                                          │ yes             │
  │                                                                          ▼                 │
  │                                                           AlertManager.raise_alert()        │
  │                                                            ├─ allowed for this client?      │
  │                                                            ├─ 5 s cooldown per camera       │
  │                                                            ├─ save JPG → violations/...     │
  │                                                            ├─ insert report → MongoDB       │
  │                                                            └─ send email (Gmail SMTP)       │
  │  annotated frame ──► latest_frame (read by Dashboard)                                       │
  └────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-step flow

| # | What happens | Where |
|---|---|---|
| 1 | Parse CLI args (`--no-gui`, custom config paths). | [scripts/run_pipeline.py](scripts/run_pipeline.py) |
| 2 | Load `app.yaml`, `model.yaml`, `cameras.yaml`. Values in `.env` (`MONGO_URI`, `MONGO_DB`, `CLIENT_NAME`, `MODEL_PATH`, `API_PORT`) **override** the YAML. | [api/config.py](api/config.py) |
| 3 | Detect hardware (CUDA / Apple MPS / Jetson / CPU) and tune OpenCV threads. | [utils/hardware.py](utils/hardware.py) |
| 4 | Connect to MongoDB (pings it; the app **stops** if Mongo is unreachable). | [api/database.py](api/database.py) |
| 5 | Orchestrator decides how many cameras can run (max = CPU cores; Jetson = cores − 1) and starts one `CameraWorker` per camera, each with its own YOLO model instance. | [pipeline/orchestrator.py](pipeline/orchestrator.py) |
| 6 | `StreamManager` reads frames in its own thread at the source FPS into a small buffer. Video files loop forever; RTSP/webcam reconnect every 3 s on failure. | [core/stream_manager.py](core/stream_manager.py) |
| 7 | Worker pulls a frame and runs YOLO (`model.track(...)` with ByteTrack when tracking is on, else `model.predict`). | [core/detector.py](core/detector.py) |
| 8 | Every detection gets a coloured box + label. `ambulance` is displayed as **"Emergency Vehicle Detected"**. | [pipeline/camera_worker.py](pipeline/camera_worker.py) |
| 9 | If the class is in `alert_classes`, `TrackMemory` counts how many frames that track ID has been seen. When it reaches `min_frames_before_alert` (default 3) it fires **once** per track ID. | [core/tracker.py](core/tracker.py) |
| 10 | `AlertManager` checks the client's allowed violation types, applies a 5-second cooldown per camera, saves the JPG, writes the Mongo report and emails it. | [core/alert_manager.py](core/alert_manager.py), [utils/email_utils.py](utils/email_utils.py) |
| 11 | Dashboard grabs the latest annotated frame + stats from every worker and draws a full-screen grid (header with total alerts, per-camera FPS/alerts footer, green/red connection dot). Press **Q** or **Esc** to quit. | [gui/dashboard.py](gui/dashboard.py) |
| 12 | Meanwhile the Flask API serves the reports from MongoDB. | [api/routes.py](api/routes.py) |
| 13 | Ctrl+C / SIGTERM / Q → stop all workers and streams, close MongoDB. | [scripts/run_pipeline.py](scripts/run_pipeline.py) |

---

## 3. Folder structure

```
Detection_application/
├── scripts/run_pipeline.py     # ENTRY POINT – starts everything
├── pipeline/
│   ├── orchestrator.py         # starts/stops one worker per camera, gives frames+stats to GUI
│   └── camera_worker.py        # per-camera loop: read → detect → track → alert → annotate
├── core/
│   ├── stream_manager.py       # threaded OpenCV capture (RTSP / video / webcam)
│   ├── detector.py             # YOLOv8 wrapper, returns Detection objects
│   ├── tracker.py              # TrackMemory (N-frame confirmation) + tracker yaml lookup
│   └── alert_manager.py        # validation, cooldown, save image, Mongo insert, email
├── api/
│   ├── config.py               # loads YAML + .env into dataclasses
│   ├── database.py             # MongoDB singleton connection
│   ├── routes.py               # Flask REST endpoints (/api/v1/...)
│   └── models.py               # ReportDocument dataclass (reference schema, not used at runtime)
├── gui/dashboard.py            # OpenCV full-screen multi-camera grid
├── utils/
│   ├── logger.py               # console + rotating file log (logs/app.log, 10 MB × 5)
│   ├── hardware.py             # device detection, worker count, OpenCV tuning
│   ├── image_utils.py          # base64 encode, draw_box, resize, crop
│   └── email_utils.py          # HTML alert email + Gmail SMTP sender
├── configs/
│   ├── app.yaml                # MongoDB, client name, API, GUI settings
│   ├── model.yaml              # model path, thresholds, alert classes, tracking, colours
│   └── cameras.yaml            # list of camera sources
├── carfall_runs/               # YOLOv8n training run "car_fall" (weights/best.pt, metrics, plots)
├── violations/                 # OUTPUT: saved alert snapshots (created at runtime)
├── logs/                       # OUTPUT: app.log (created at runtime)
├── classes.txt                 # training class list
├── Vehicle_Detection_KT_Documentation.pdf  # how the model was trained
├── .env / .env.example         # secrets & overrides
└── requirements.txt
```

> `Project Structure.txt` is the original design plan; some files it lists (`gui/widgets.py`, `scripts/run_gui.py`, `scripts/test_camera.py`) were never created.

---

## 4. Setup

### 4.1 Requirements
- Python 3.9+
- MongoDB running locally or remotely
- The YOLO weights file (e.g. `Vehicle_dect_may1.pt`, **not included in this folder**) – or use `carfall_runs/runs/detect/car_fall/weights/best.pt`
- NVIDIA GPU optional (falls back to CPU)

### 4.2 Install
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
> ⚠️ `requirements.txt` lists `bson`. That PyPI package **conflicts with the `bson` bundled in `pymongo`** and can break imports. If you see `ImportError` around `bson`/`pymongo`, run `pip uninstall -y bson && pip install --force-reinstall pymongo`.

### 4.3 Configure `.env`
```bash
cp .env.example .env
```
| Variable | Meaning |
|---|---|
| `MONGO_URI` | e.g. `mongodb://localhost:27017/` |
| `MONGO_DB` | database name (default `my_project`) |
| `CLIENT_NAME` | must match `clientName` of a document in the `clients` collection |
| `MODEL_PATH` | path to the `.pt` weights (overrides `model.yaml`) |
| `API_PORT` | REST API port (default `5050`) |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` ... |

### 4.4 Prepare MongoDB (important)
The app looks up the client **once at startup**:
```js
// mongosh
use my_project
db.clients.insertOne({
  clientName: "Biswa",
  violationTypes: ["Emergency Vehicle Detected"]   // names that are allowed to raise alerts
})
```
- If the client doc is **missing**, reports are still saved but with `client: null`, and no type filtering is applied.
- If `violationTypes` is set, alert names **must match exactly**. Note the ambulance class is renamed to `"Emergency Vehicle Detected"` before this check; any other alert class uses its raw YOLO name (e.g. `upside_down`).

### 4.5 Add cameras – `configs/cameras.yaml`
```yaml
cameras:
  - id: cam_01            # unique id
    name: "Gate_01"       # shown on screen + used as folder name for snapshots
    location: "Gate_01"
    source: "rtsp://user:pass@192.168.1.101:554/stream"
    type: rtsp            # rtsp | video | webcam
  - id: cam_02
    name: "Navalur"
    location: "Navalur"
    source: "test.mp4"    # relative to the folder you run from
    type: video           # video files loop automatically
  - id: cam_03
    name: "Parking_01"
    location: "Parking_01"
    source: 0             # webcam index
    type: webcam
```

### 4.6 Model & alert settings – `configs/model.yaml`
| Key | Meaning |
|---|---|
| `model.path` | weights file (overridden by `MODEL_PATH`) |
| `conf_threshold` / `iou_threshold` | YOLO confidence / NMS IoU |
| `imgsz` | inference size (640) |
| `device` | `auto`, `cuda`, `cpu`, `mps` |
| `half` | FP16 – only applied on CUDA |
| `alert_classes` | YOLO class names that raise alerts (e.g. `ambulance`, `upside_down`) |
| `tracking.enabled` / `method` | `bytetrack` (default) or `botsort` |
| `tracking.min_frames_before_alert` | frames a track must be seen before alerting |
| `colors` | BGR box colour per class |

`all_classes` in this file is informational only – the real class names come from the model file itself.

---

## 5. Run

**Always run from the project root** (config, `logs/` and `violations/` paths are relative to the current folder):

```bash
cd Detection_application
python scripts/run_pipeline.py            # with full-screen dashboard (Q / Esc to quit)
python scripts/run_pipeline.py --no-gui   # headless – servers / Jetson (Ctrl+C to stop)

# custom config files
python scripts/run_pipeline.py --config configs/app.yaml \
       --model-config configs/model.yaml --cameras-config configs/cameras.yaml
```

---

## 6. Outputs

**Snapshot files** – `violations/YYYY-MM-DD/<camera name>/HH-MM-SS_<Violation>_<trackId>.jpg`

**MongoDB `reports` document**
```json
{
  "client": "ObjectId(...)",
  "camname": "Gate_02",
  "location": "Gate_02",
  "violationType": "Emergency Vehicle Detected",
  "tags": "Violation",
  "assigned": "Unassigned",
  "status": "Open",
  "imagepath": "data:image/jpeg;base64,...",
  "localImagePath": "violations/2026-10-28/Gate_02/22-12-47_EmergencyVehicleDetected_7.jpg",
  "bbox": { "x1": 0, "y1": 0, "x2": 0, "y2": 0, "width": 0, "height": 0 },
  "trackId": 7,
  "confidence": 0.91,
  "comments": [],
  "createdAt": "UTC datetime",
  "updatedAt": "UTC datetime"
}
```

**Email** – HTML alert with location, camera, event, time and the snapshot attached (sender/recipient currently set in code, see §9).

**Logs** – `logs/app.log`

---

## 7. REST API (`http://<host>:5050/api/v1`)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/reports?page=1&limit=20&camname=Gate_01&status=Open&client_id=<id>` | paginated reports, newest first |
| GET | `/reports/<id>` | one report |
| PATCH | `/reports/<id>/status` | body `{"status": "Closed"}` |
| GET | `/clients` | all clients |

```bash
curl http://localhost:5050/api/v1/reports?limit=5
curl -X PATCH -H "Content-Type: application/json" \
     -d '{"status":"Closed"}' http://localhost:5050/api/v1/reports/<id>/status
```

---

## 8. The model

- YOLOv8n, 640 px, 7 classes: `car, truck, bike, bus, others, ambulance, upside_down` (see [classes.txt](classes.txt)).
- Training procedure (labelling, 80/20 split, hyper-parameters) is in [Vehicle_Detection_KT_Documentation.pdf](Vehicle_Detection_KT_Documentation.pdf).
- `carfall_runs/runs/detect/car_fall/` holds a 120-epoch training run (val mAP50 ≈ 0.99, mAP50-95 ≈ 0.92) with `weights/best.pt` and all plots.

**To alert on a new class** (e.g. overturned vehicles): add its YOLO name to `alert_classes` in `model.yaml`, and add the same name to the client's `violationTypes` in MongoDB.

---

## 9. Known issues / things to fix

| Issue | Where | Suggested fix |
|---|---|---|
| Gmail sender + app password and recipient email are **hard-coded** | `utils/email_utils.py`, `core/alert_manager.py` | Move to `.env` (`SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL_TO`) and rotate the exposed password |
| Email is sent synchronously inside the camera loop → video stalls for a few seconds per alert | `core/alert_manager.py` | Send in a background thread/queue |
| `deepsort` maps to `deepocsort.yaml`, which Ultralytics does not ship | `core/tracker.py` | Use `bytetrack` or `botsort` |
| Detections without a track ID all share ID `-1`, so only the first one ever alerts | `pipeline/camera_worker.py` | Handle untracked detections separately |
| Cooldown (5 s) is per camera, not per vehicle, and not configurable | `core/alert_manager.py` | Make it a config value |
| Dashboard assumes a 1920×1080 screen | `gui/dashboard.py` | Read actual screen size |
| Running from `scripts/` creates `scripts/logs` and `scripts/violations` | — | Run from project root |
| `bson` in `requirements.txt` conflicts with `pymongo` | `requirements.txt` | Remove the `bson` line |
| `pipeline/camera_worker (copy).py` is an old backup | — | Delete when no longer needed |
