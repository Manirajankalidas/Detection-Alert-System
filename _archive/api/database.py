from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from utils.logger import get_logger
from typing import Optional

logger = get_logger("database")

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def init_db(uri: str, db_name: str) -> Database:
    """
    Initialize MongoDB connection (singleton pattern).
    Call once at app startup.
    """
    global _client, _db
    if _db is not None:
        return _db
    try:
        _client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            maxPoolSize=50,
        )
        _client.admin.command("ping")
        _db = _client[db_name]
        logger.info(f"MongoDB connected: {uri} | DB: {db_name}")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise
    return _db


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db


def get_collection(name: str) -> Collection:
    return get_db()[name]


def close_db():
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")
