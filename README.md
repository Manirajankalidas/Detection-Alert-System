# Detection Alert System

A web app that watches cameras or videos with a **YOLOv8** model and **raises an alert** when an alert class (e.g. an ambulance) appears.
Each alert shows up live on the dashboard and is saved to a local folder. There's no database; everything is JSON files and images.

---

## Run

```bash
pip install -r requirements.txt
python app.py
```
Open **http://localhost:8000**

| Option | Default | Meaning |
|---|---|---|
| `--port` | `8000` | web port. If it's busy, the next free port is used (8001, 8002, ...). The log shows the actual URL. |
| `--host` | `0.0.0.0` | other PCs on the network can open `http://<this-pc-ip>:8000` |
| `--data-dir` | `data` | where settings, alerts and images are saved |
| `--no-autostart` | off | don't start detection when the app starts |

---

## Use it in 3 steps

1. **Cameras** tab → **+ Add camera**
   - *Name*, *Location*
   - *Type*: `Video file` / `RTSP stream` / `Webcam`
   - *Source*: file path (e.g. `/home/me/videos/road.mp4`), RTSP URL, or webcam number (`0`)
   - Click **Test connection** to check it, then **Save**.
2. **Model** tab
   - *Model path*: defaults to `models/car_fall.pt`
   - *Image size*, *Device* (auto / cpu / cuda), *Detection confidence*, *NMS IoU*, *Tracking*
   - **Alert rules**:
     - *Alert confidence threshold*: how sure the model must be before a detection counts
     - *Confirm after N frames*: the object must be seen N frames in a row (stops flicker false alarms)
     - *Cooldown*: minimum seconds between two alerts of the same class on one camera
   - **Classes** → **Load classes from model file**. For each class, set its **colour**, **Show** (draw it), **Alert** (raise alerts) and **Alert label** (e.g. `ambulance` → `Emergency Vehicle Detected`).
   - **Save & apply**
3. **Dashboard**: live video of every camera with boxes. New alerts pop up on the right with a sound and a red flash on the camera tile. Click a thumbnail for the full image; ✓ acknowledges, ✕ deletes.
   - **Big view:** **double-click a camera** (or click ⤢) to open it large at full resolution. The alert panel then shows only that camera. Return with **← All cameras**, a double-click, **Esc** or the browser **Back** button.
4. **Violations**: a table of every detected violation with its image, date/time, camera, location, type, confidence, track ID and status.
   - Filter by camera, violation type, status (New / Acknowledged) and date range.
   - 25 / 50 / 100 rows per page.
   - **View**, **Acknowledge** or **Delete** each row, or **Acknowledge all**.
   - **Export CSV** downloads the rows matching the current filters, e.g. for analysis in Excel.
   - The red number on the tab is the count of new (unacknowledged) violations.

Saving any setting automatically restarts detection if it's running. **Start / Stop** is in the top-right corner.

---

## Folder structure

```
Detection_application/
├── app.py            ← RUN THIS. Web server + all page/API routes
├── engine.py         ← video reading, YOLO detection, tracking, alert rules (one thread per camera)
├── storage.py        ← saves settings and alerts as JSON + images
├── templates/
│   └── index.html    ← the web page
├── static/
│   ├── app.js        ← page logic (tabs, forms, live video, live alerts)
│   └── style.css
├── models/
│   └── car_fall.pt   ← trained YOLOv8n model (car, truck, bike, bus, others, ambulance, car_fall)
├── data/             ← created on first run (see below)
├── docs/training/    ← how the model was trained (KT PDF) + training results/plots
├── requirements.txt
└── _archive/         ← the old MongoDB/OpenCV version, kept for reference only (safe to delete)
```

### What gets saved (`data/`)

```
data/
├── cameras.json            camera list
├── model.json              model + alert settings
├── alerts.jsonl            one alert per line (JSON)
├── app.log                 log file
└── alerts/2026-11-01/Gate_02/
    ├── 14-03-22-415_ambulance_7.jpg         full frame with the alert box
    └── 14-03-22-415_ambulance_7_thumb.jpg   small preview
```

One line of `alerts.jsonl`:
```json
{"id": "8e92d34727eb", "time": "2026-11-01T14:03:22", "camera_id": "gate_02", "camera_name": "Gate_02",
 "location": "Main gate", "class_name": "ambulance", "label": "Emergency Vehicle Detected",
 "confidence": 0.91, "track_id": 7, "bbox": {"x1": 151, "y1": 60, "x2": 900, "y2": 700},
 "image": "2026-11-01/Gate_02/14-03-22-415_ambulance_7.jpg",
 "thumb": "2026-11-01/Gate_02/14-03-22-415_ambulance_7_thumb.jpg", "status": "new"}
```
To start fresh, stop the app and delete `data/`.

---

## How it works

```
              ┌──────────────────────────── engine.py (per camera) ─────────────────────────────┐
 video / RTSP │ VideoStream ──frame──► Detector (YOLO + ByteTrack) ──► draw boxes ──► alert class?│
 / webcam ───►│ (reads at real-time       │                                             │ yes      │
              │  speed, loops videos,     │                          seen N frames in a row,     │
              │  reconnects cameras)      │                          new track, cooldown over?   │
              │                           ▼                                             │ yes      │
              │                  latest annotated frame                    storage.AlertStore.add │
              └───────────────────────────┼─────────────────────────────────────────────┼────────┘
                                          │                                             │ saves image + alerts.jsonl
                                          ▼                                             ▼
 browser ◄── app.py ── /api/cameras/<id>/frame.jpg (live video)      /api/alerts/events (instant push)
```

- **Alert rules** (in `CameraRunner._process`): a detection counts only if its class is ticked *Alert* and its confidence is above the alert threshold. It must then be seen in *N* consecutive frames. With tracking on, each vehicle (track ID) alerts **once**. The *cooldown* stops repeated alerts of the same class.
- Each camera has its **own copy of the model**, because tracking state is per camera.

---

## API (optional, for scripts)

| Method | Endpoint | |
|---|---|---|
| GET / POST | `/api/cameras` | list / add camera |
| PUT / DELETE | `/api/cameras/<id>` | edit / delete camera |
| POST | `/api/cameras/test` | try a source, returns one frame |
| GET | `/api/cameras/<id>/frame.jpg` | latest annotated frame (`?full=1` for full resolution) |
| GET / PUT | `/api/model` | read / save model + alert settings |
| POST | `/api/model/classes` | `{"path": "..."}` → class names in a `.pt` |
| GET | `/api/engine/status` | per-camera status / FPS / alert counts |
| POST | `/api/engine/start`, `/api/engine/stop` | control detection |
| GET | `/api/alerts?limit=&offset=&camera=&class=&status=&from=&to=` | alerts, newest first, plus `total` matching (dates `YYYY-MM-DD`) |
| GET | `/api/alerts/export.csv?camera=&class=&status=&from=&to=` | download matching alerts as CSV |
| PATCH | `/api/alerts/<id>` | `{"status": "acknowledged"}` |
| POST | `/api/alerts/acknowledge-all` | |
| DELETE | `/api/alerts/<id>`, `/api/alerts` | delete one / all (records + images) |
| GET | `/api/alerts/events` | live stream of new alerts (Server-Sent Events) |

---

## The model

- YOLOv8n, 640 px, trained for 120 epochs (val mAP50 ≈ 0.99, mAP50-95 ≈ 0.92). The plots and metrics are in `docs/training/car_fall_run/`.
- Classes: `car, truck, bike, bus, others, ambulance, car_fall`.
- The training procedure is in `docs/training/Vehicle_Detection_KT_Documentation.pdf`.
- To use another model, put the `.pt` in `models/`, set its path on the Model tab, and click **Load classes from model file**.

## Notes

- This uses Flask's built-in server with **no login**. It's fine for a lab PC or local network; don't expose it to the internet.
- On CPU expect roughly 5-8 FPS per 1080p camera. Set *Device* to `cuda` on a GPU machine for more cameras.
- Relative paths (videos, models) are resolved from this folder.
