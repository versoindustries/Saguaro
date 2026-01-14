"""
Chronicle Storage: Persistence for Temporal Indexing
"""

import sqlite3
import time
import json
import logging
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any

logger = logging.getLogger("saguaro.chronicle.storage")


class ChronicleStorage:
    def __init__(self, db_path: str = ".saguaro/chronicle.db"):
        self.db_path = db_path
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Snapshots table store point-in-time semantic states
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    commit_hash TEXT,
                    description TEXT,
                    hd_state_blob BLOB,
                    metadata TEXT CHECK(json_valid(metadata) OR metadata IS NULL)
                )
            """)
            # Drift logs store calculated divergence between snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    snapshot_id_a INTEGER,
                    snapshot_id_b INTEGER,
                    drift_score REAL,
                    details TEXT,
                    FOREIGN KEY(snapshot_id_a) REFERENCES snapshots(id),
                    FOREIGN KEY(snapshot_id_b) REFERENCES snapshots(id)
                )
            """)
            conn.commit()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def save_snapshot(
        self,
        hd_state_blob: bytes,
        commit_hash: str = "HEAD",
        description: str = "",
        metadata: Dict[str, Any] = None,
    ) -> int:
        """
        Save a new semantic snapshot of the codebase.

        Args:
            hd_state_blob: Serialized bytes of the Hyperdimensional state (Time Crystal)
            commit_hash: Git commit hash associated with this state
            description: Human readable description
            metadata: Additional JSON metadata

        Returns:
            id of the inserted snapshot
        """
        timestamp = time.time()
        metadata_json = json.dumps(metadata) if metadata else None

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO snapshots (timestamp, commit_hash, description, hd_state_blob, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (timestamp, commit_hash, description, hd_state_blob, metadata_json),
            )
            snapshot_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Saved snapshot {snapshot_id} for commit {commit_hash}")
            return snapshot_id

    def get_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def log_drift(
        self,
        snapshot_id_a: int,
        snapshot_id_b: int,
        drift_score: float,
        details: str = "",
    ):
        timestamp = time.time()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO drift_logs (timestamp, snapshot_id_a, snapshot_id_b, drift_score, details)
                VALUES (?, ?, ?, ?, ?)
            """,
                (timestamp, snapshot_id_a, snapshot_id_b, drift_score, details),
            )
            conn.commit()
