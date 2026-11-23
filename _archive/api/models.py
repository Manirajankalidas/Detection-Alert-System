from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from bson import ObjectId


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ReportDocument:
    """
    Mirrors the MongoDB reports collection schema exactly.
    """
    client_id: str               # ObjectId string of client
    camname: str
    location: str
    violation_type: str
    tags: str = "Violation"
    assigned: str = "Unassigned"
    status: str = "Open"
    imagepath: str = ""          # data:image/jpeg;base64,...
    comments: List[str] = field(default_factory=list)
    track_id: Optional[int] = None
    confidence: Optional[float] = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def to_mongo_dict(self) -> dict:
        return {
            "client": ObjectId(self.client_id),
            "camname": self.camname,
            "location": self.location,
            "violationType": self.violation_type,
            "tags": self.tags,
            "assigned": self.assigned,
            "status": self.status,
            "imagepath": self.imagepath,
            "comments": self.comments,
            "trackId": self.track_id,
            "confidence": round(self.confidence, 4) if self.confidence else None,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
