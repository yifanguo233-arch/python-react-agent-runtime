from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, default=str)


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


@dataclass
class RunSummary:
    run_id: str
    task: str
    project_directory: str
    target_project: str | None
    started_at: str
    finished_at: str | None
    status: str
    model: str
    log_path: str
    final_answer: str
    event_count: int
    tool_count: int
    approval_count: int
    avg_tool_latency_ms: float | None


class TraceStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    project_directory TEXT NOT NULL,
                    target_project TEXT,
                    model TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    final_answer TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    tool_name TEXT,
                    latency_ms REAL,
                    human_approval INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_events_run_seq
                    ON events(run_id, seq);
                CREATE INDEX IF NOT EXISTS idx_runs_started_at
                    ON runs(started_at DESC);
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "target_project" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN target_project TEXT")

    def start_run(
        self,
        task: str,
        project_directory: str,
        model: str,
        log_path: str,
        *,
        target_project: str | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, task, project_directory, target_project, model, log_path, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, task, project_directory, target_project, model, log_path, utc_now()),
            )
        return run_id

    def add_event(
        self,
        run_id: str,
        event_type: str,
        content: str,
        *,
        label: str = "",
        tool_name: str | None = None,
        latency_ms: float | None = None,
        human_approval: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            seq = int(row["next_seq"])
            conn.execute(
                """
                INSERT INTO events (
                    run_id, seq, event_type, label, content, tool_name,
                    latency_ms, human_approval, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    seq,
                    event_type,
                    label,
                    content,
                    tool_name,
                    latency_ms,
                    None if human_approval is None else int(bool(human_approval)),
                    _json_dumps(metadata),
                    utc_now(),
                ),
            )

    def finish_run(self, run_id: str, final_answer: str, status: str = "completed") -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET finished_at = ?, status = ?, final_answer = ?
                WHERE run_id = ?
                """,
                (utc_now(), status, final_answer, run_id),
            )

    def list_runs(self, limit: int = 20, project: str | None = None) -> list[RunSummary]:
        filters = ""
        params: list[Any] = []
        if project:
            filters = "WHERE COALESCE(r.target_project, r.project_directory) = ?"
            params.append(project)
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    r.*,
                    COUNT(e.id) AS event_count,
                    SUM(CASE WHEN e.event_type = 'tool_result' THEN 1 ELSE 0 END) AS tool_count,
                    SUM(CASE WHEN e.event_type = 'human_approval'
                              OR (e.event_type = 'tool_policy' AND e.human_approval IS NOT NULL)
                             THEN 1 ELSE 0 END) AS approval_count,
                    AVG(CASE WHEN e.latency_ms IS NOT NULL THEN e.latency_ms END) AS avg_tool_latency_ms
                FROM runs r
                LEFT JOIN events e ON e.run_id = r.run_id
                {filters}
                GROUP BY r.run_id
                ORDER BY r.started_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_summary_from_row(row) for row in rows]

    def latest_run_id(self, project: str | None = None) -> str | None:
        query = "SELECT run_id FROM runs"
        params: tuple[Any, ...] = ()
        if project:
            query += " WHERE COALESCE(target_project, project_directory) = ?"
            params = (project,)
        query += " ORDER BY started_at DESC LIMIT 1"
        with self.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return str(row["run_id"]) if row else None

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def get_events(self, run_id: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()


def format_run_list(summaries: list[RunSummary]) -> str:
    if not summaries:
        return "No runs recorded yet."
    lines = ["Recent runs:"]
    for item in summaries:
        latency = "-" if item.avg_tool_latency_ms is None else f"{item.avg_tool_latency_ms:.1f}ms"
        task = _one_line(item.task, 70)
        runtime = _one_line(item.project_directory, 45)
        target = _one_line(item.target_project or item.project_directory, 50)
        lines.append(
            f"- {item.run_id} [{item.status}] {item.started_at} "
            f"events={item.event_count} tools={item.tool_count} approvals={item.approval_count} "
            f"avg_tool_latency={latency} runtime={runtime} target={target} task={task}"
        )
    return "\n".join(lines)


def format_run_report(store: TraceStore, run_id: str) -> str:
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")
    events = store.get_events(run_id)
    lines = [
        f"run_id={run['run_id']}",
        f"status={run['status']}",
        f"started_at={run['started_at']}",
        f"finished_at={run['finished_at'] or '-'}",
        f"model={run['model']}",
        f"runtime_project={run['project_directory']}",
        f"target_project={run['target_project'] or run['project_directory']}",
        f"log_path={run['log_path']}",
        "",
        "Task:",
        str(run["task"]),
    ]

    plans = [event for event in events if event["event_type"] == "plan"]
    if plans:
        lines.extend(["", "Plan:"])
        lines.append(plans[-1]["content"])

    lines.extend(["", "Trace:"])
    for event in events:
        event_type = event["event_type"]
        if event_type == "plan":
            continue
        content = str(event["content"]).strip()
        prefix = f"{int(event['seq']):03d} {event_type}"
        if event["tool_name"]:
            prefix += f" tool={event['tool_name']}"
        if event["latency_ms"] is not None:
            prefix += f" latency={float(event['latency_ms']):.1f}ms"
        if event["human_approval"] is not None:
            approved = "yes" if int(event["human_approval"]) else "no"
            prefix += f" human_approval={approved}"
        lines.append(f"{prefix}: {_one_line(content, 180)}")

    evidence = [event for event in events if event["event_type"] == "evidence_ledger"]
    if evidence:
        lines.extend(["", "Evidence Ledger:"])
        lines.append(evidence[-1]["content"].strip())

    if run["final_answer"]:
        lines.extend(["", "Final Answer:", str(run["final_answer"])])
    return "\n".join(lines)


def _summary_from_row(row: sqlite3.Row) -> RunSummary:
    return RunSummary(
        run_id=str(row["run_id"]),
        task=str(row["task"]),
        project_directory=str(row["project_directory"]),
        target_project=(str(row["target_project"]) if row["target_project"] else None),
        started_at=str(row["started_at"]),
        finished_at=row["finished_at"],
        status=str(row["status"]),
        model=str(row["model"]),
        log_path=str(row["log_path"]),
        final_answer=str(row["final_answer"]),
        event_count=int(row["event_count"] or 0),
        tool_count=int(row["tool_count"] or 0),
        approval_count=int(row["approval_count"] or 0),
        avg_tool_latency_ms=None if row["avg_tool_latency_ms"] is None else float(row["avg_tool_latency_ms"]),
    )


def _one_line(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
