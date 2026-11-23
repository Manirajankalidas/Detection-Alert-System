from datetime import datetime
from typing import Optional
from bson import ObjectId

from api.database import get_collection
from utils.image_utils import encode_frame_to_base64
from utils.logger import get_logger
from utils.email_utils import send_alert_email, build_email_html

import numpy as np
import os
import cv2
import time

logger = get_logger("alert_manager")


class AlertManager:

    def __init__(self, client_name: str, reports_collection: str, clients_collection: str):
        self.reports_col = get_collection(reports_collection)
        self.clients_col = get_collection(clients_collection)

        self.client_doc = self.clients_col.find_one({"clientName": client_name})
        self.client_id = self.client_doc["_id"] if self.client_doc else None

        self.allowed_violations = self.client_doc.get("violationTypes", []) if self.client_doc else []

        self.last_alert_time = {}
        self.cooldown_seconds = 5

    def save_violation_image(self, frame, cam_name, violation, track_id, confidence, x1, y1, x2, y2):
        now = datetime.now()

        folder = f"violations/{now.strftime('%Y-%m-%d')}/{cam_name}"
        os.makedirs(folder, exist_ok=True)

        label = f"{violation} | ID:{track_id} | {confidence*100:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        path = f"{folder}/{now.strftime('%H-%M-%S')}_{violation.replace(' ', '')}_{track_id}.jpg"
        cv2.imwrite(path, frame)

        return path

    def raise_alert(
        self,
        frame: np.ndarray,
        cam_name: str,
        location: str,
        class_name: str,
        track_id: Optional[int],
        confidence: float,
        x1: int, y1: int, x2: int, y2: int,
        save_crop: bool = False,
    ) -> Optional[str]:

        try:
            print("ALERT TRIGGERED FOR:", class_name)

            # ============================
            # ✅ VALIDATION
            # ============================
            if self.allowed_violations and class_name not in self.allowed_violations:
                print("❌ SKIPPED (NOT IN DB):", class_name)
                print("Allowed:", self.allowed_violations)
                return None

            # ============================
            # ✅ COOLDOWN
            # ============================
            now_time = time.time()
            last_time = self.last_alert_time.get(cam_name, 0)

            if now_time - last_time < self.cooldown_seconds:
                return None

            self.last_alert_time[cam_name] = now_time

            # ============================
            # 🔥 DRAW BBOX FOR DASHBOARD
            # ============================
            annotated = frame.copy()

            label = f"{class_name} | ID:{track_id} | {confidence*100:.0f}%"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            # ============================
            # ✅ SAVE LOCAL IMAGE
            # ============================
            local_path = self.save_violation_image(
                frame.copy(),
                cam_name,
                class_name,
                track_id,
                confidence,
                x1, y1, x2, y2
            )

            # ============================
            # 🔥 BASE64 WITH BBOX
            # ============================
            image_b64 = encode_frame_to_base64(annotated)

            # ============================
            # 🔥 BBOX DATA
            # ============================
            bbox = {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            }

            # ============================
            # ✅ SAVE TO MONGO
            # ============================
            data = {
                "client": ObjectId(self.client_id) if self.client_id else None,
                "camname": cam_name,
                "location": location,
                "violationType": class_name,
                "tags": "Violation",
                "assigned": "Unassigned",
                "status": "Open",

                "imagepath": image_b64,
                "localImagePath": local_path,
                "bbox": bbox,

                "trackId": track_id,
                "confidence": float(confidence),

                "comments": [],
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
            }

            result = self.reports_col.insert_one(data)
            print("✅ ALERT SAVED:", result.inserted_id)

            # ============================
            # 📧 SEND EMAIL ALERT
            # ============================
            try:
                html = build_email_html(
                    cam_name=cam_name,
                    location=location,
                    violation=class_name
                )

                send_alert_email(
                    to_email="biswa@quantictech.ai",  #CHANGE THIS
                    subject=f"🚨 Alert: {class_name}",
                    html_content=html,
                    image_path=local_path
                )

            except Exception as e:
                print("❌ Email failed:", e)

            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"❌ Alert failed: {e}")
            return None