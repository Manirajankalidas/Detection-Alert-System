from flask import Blueprint, jsonify, request
from bson import ObjectId, json_util
from api.database import get_collection
from utils.logger import get_logger
import json

logger = get_logger("routes")
api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


def _serialize(doc) -> dict:
    """Convert MongoDB document to JSON-serializable dict."""
    return json.loads(json_util.dumps(doc))


@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "alert-system"}), 200


@api_bp.route("/reports", methods=["GET"])
def get_reports():
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 20))
        client_id = request.args.get("client_id")
        cam_name = request.args.get("camname")
        status = request.args.get("status")
        skip = (page - 1) * limit

        query = {}
        if client_id:
            query["client"] = ObjectId(client_id)
        if cam_name:
            query["camname"] = cam_name
        if status:
            query["status"] = status

        col = get_collection("reports")
        total = col.count_documents(query)
        docs = list(col.find(query).sort("createdAt", -1).skip(skip).limit(limit))

        return jsonify({
            "success": True,
            "total": total,
            "page": page,
            "limit": limit,
            "data": [_serialize(d) for d in docs]
        }), 200
    except Exception as e:
        logger.error(f"GET /reports error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/reports/<report_id>", methods=["GET"])
def get_report(report_id: str):
    try:
        doc = get_collection("reports").find_one({"_id": ObjectId(report_id)})
        if not doc:
            return jsonify({"success": False, "error": "Not found"}), 404
        return jsonify({"success": True, "data": _serialize(doc)}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/reports/<report_id>/status", methods=["PATCH"])
def update_report_status(report_id: str):
    try:
        body = request.get_json()
        new_status = body.get("status")
        if not new_status:
            return jsonify({"success": False, "error": "status required"}), 400

        col = get_collection("reports")
        result = col.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": {"status": new_status}}
        )
        if result.matched_count == 0:
            return jsonify({"success": False, "error": "Not found"}), 404
        return jsonify({"success": True, "message": "Status updated"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/clients", methods=["GET"])
def get_clients():
    try:
        docs = list(get_collection("clients").find())
        return jsonify({"success": True, "data": [_serialize(d) for d in docs]}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
