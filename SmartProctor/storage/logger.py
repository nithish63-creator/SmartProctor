import os
import sqlite3
import time
import json
import threading

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STORAGE_DIR = os.path.join(BASE_DIR, 'storage_files')
DB_PATH = os.path.join(STORAGE_DIR, 'session.db')
EVIDENCE_DIR = os.path.join(STORAGE_DIR, 'evidence')

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time REAL,
    end_time REAL,
    student_name TEXT,
    exam_name TEXT,
    duration REAL
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    event_type TEXT,
    details TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    severity TEXT,
    reason TEXT,
    evidence_path TEXT,
    source_event_id INTEGER,
    FOREIGN KEY(source_event_id) REFERENCES logs(id)
);
"""


class ExamLogger:
    def __init__(self, db_path=None):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        os.makedirs(EVIDENCE_DIR, exist_ok=True)

        self.db_path = db_path if db_path else DB_PATH
        self.lock = threading.Lock()

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.executescript(DB_SCHEMA)
        self.conn.commit()

    # ---------- Logging Methods ----------
    def log(self, event_type, details, commit=True):
        """
        General-purpose log function.
        Returns the log ID for potential cross-linking.
        """
        if not details:
            details = {}
        if not isinstance(details, str):
            details = json.dumps(details)

        with self.lock:
            for _ in range(3):  # retry up to 3 times
                try:
                    cur = self.conn.cursor()
                    cur.execute(
                        "INSERT INTO logs (ts, event_type, details) VALUES (?,?,?)",
                        (time.time(), event_type, details)
                    )
                    if commit:
                        self.conn.commit()
                    return cur.lastrowid
                except sqlite3.OperationalError:
                    time.sleep(0.05)
            print("[ExamLogger] ⚠️ Failed to insert log after retries.")
        return None

    def incident(self, severity, reason, evidence_path=None, extra=None, source_event_id=None):
        """
        Insert an incident into DB.
        severity: 'low' | 'medium' | 'high'
        reason: e.g. 'cell_phone_detected', 'speech_detected'
        evidence_path: optional path to saved image/audio
        extra: dict for additional info
        source_event_id: optional log_id for linking
        """
        try:
            details = {"reason": reason}
            if extra:
                details["extra"] = extra

            with self.lock:
                cur = self.conn.cursor()
                cur.execute(
                    "INSERT INTO incidents (ts, severity, reason, evidence_path, source_event_id) VALUES (?,?,?,?,?)",
                    (time.time(), severity, reason, evidence_path, source_event_id)
                )
                self.conn.commit()

            # Also mirror to logs for traceability
            self.log("incident", details, commit=False)
            print(f"[ExamLogger] Incident logged: {reason} ({severity})")

        except Exception as e:
            print(f"[ExamLogger] Error logging incident: {e}")

    # ---------- Session Metadata ----------
    def start_session(self, student_name=None, exam_name=None):
        """Record session start info."""
        cur = self.conn.cursor()
        start_time = time.time()
        cur.execute(
            "INSERT INTO session_meta (start_time, student_name, exam_name) VALUES (?,?,?)",
            (start_time, student_name, exam_name)
        )
        self.conn.commit()
        print(f"[ExamLogger] Session started for {student_name or 'Unknown'} - {exam_name or 'Exam'}")
        return cur.lastrowid

    def end_session(self):
        """Mark session end time."""
        cur = self.conn.cursor()
        end_time = time.time()
        cur.execute("SELECT id, start_time FROM session_meta ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            sid, start = row
            duration = end_time - start
            cur.execute(
                "UPDATE session_meta SET end_time=?, duration=? WHERE id=?",
                (end_time, duration, sid)
            )
            self.conn.commit()
            print(f"[ExamLogger] Session ended (duration {duration:.1f}s)")

    # ---------- Retrieval ----------
    def list_logs(self, limit=1000):
        cur = self.conn.cursor()
        cur.execute("SELECT id, ts, event_type, details FROM logs ORDER BY ts DESC LIMIT ?", (limit,))
        return cur.fetchall()

    def list_incidents(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, ts, severity, reason, evidence_path FROM incidents ORDER BY ts ASC")
        return cur.fetchall()

    def get_session_meta(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM session_meta ORDER BY id DESC LIMIT 1")
        return cur.fetchone()

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass
