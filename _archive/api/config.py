import os
import yaml
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Any

load_dotenv()


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class MongoConfig:
    uri: str
    database: str
    reports_collection: str
    clients_collection: str


@dataclass
class ClientConfig:
    name: str


@dataclass
class ModelConfig:
    path: str
    conf_threshold: float
    iou_threshold: float
    imgsz: int
    device: str
    half: bool
    alert_classes: List[str]
    tracking_enabled: bool
    tracking_method: str
    min_frames_before_alert: int
    colors: Dict[str, List[int]]


@dataclass
class CameraConfig:
    id: str
    name: str
    location: str
    source: Any
    type: str


@dataclass
class AppConfig:
    mongo: MongoConfig
    client: ClientConfig
    model: ModelConfig
    cameras: List[CameraConfig]
    api_host: str
    api_port: int
    gui_enabled: bool
    gui_tile_width: int
    gui_tile_height: int
    gui_max_cols: int


def load_config(
    app_yaml: str = "configs/app.yaml",
    model_yaml: str = "configs/model.yaml",
    cameras_yaml: str = "configs/cameras.yaml",
) -> AppConfig:
    app_raw = _load_yaml(app_yaml)
    model_raw = _load_yaml(model_yaml)
    cam_raw = _load_yaml(cameras_yaml)

    mongo_raw = app_raw["mongodb"]
    mongo = MongoConfig(
        uri=os.getenv("MONGO_URI", mongo_raw["uri"]),
        database=os.getenv("MONGO_DB", mongo_raw["database"]),
        reports_collection=mongo_raw["collections"]["reports"],
        clients_collection=mongo_raw["collections"]["clients"],
    )

    client_cfg = ClientConfig(
        name=os.getenv("CLIENT_NAME", app_raw["client"]["name"])
    )

    tracking_raw = model_raw["tracking"]
    m = model_raw["model"]
    model_cfg = ModelConfig(
        path=os.getenv("MODEL_PATH", m["path"]),
        conf_threshold=m["conf_threshold"],
        iou_threshold=m["iou_threshold"],
        imgsz=m["imgsz"],
        device=m["device"],
        half=m.get("half", False),
        alert_classes=model_raw.get("alert_classes", ["ambulance"]),
        tracking_enabled=tracking_raw.get("enabled", True),
        tracking_method=tracking_raw.get("method", "bytetrack"),
        min_frames_before_alert=tracking_raw.get("min_frames_before_alert", 3),
        colors=model_raw.get("colors", {}),
    )

    cameras = [
        CameraConfig(
            id=c["id"],
            name=c["name"],
            location=c["location"],
            source=c["source"],
            type=c["type"],
        )
        for c in cam_raw["cameras"]
    ]

    gui_raw = app_raw.get("gui", {})
    api_raw = app_raw.get("api", {})

    return AppConfig(
        mongo=mongo,
        client=client_cfg,
        model=model_cfg,
        cameras=cameras,
        api_host=api_raw.get("host", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", api_raw.get("port", 5050))),
        gui_enabled=gui_raw.get("enabled", True),
        gui_tile_width=gui_raw.get("tile_width", 640),
        gui_tile_height=gui_raw.get("tile_height", 360),
        gui_max_cols=gui_raw.get("max_cols", 3),
    )
