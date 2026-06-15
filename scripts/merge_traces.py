from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_traces import TraceStore


def _resolve_db_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path / ".runs" / "traces.sqlite3"
    return path


def _load_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            run_id,
            task,
            project_directory,
            target_project,
            model,
            log_path,
            started_at,
            finished_at,
            status,
            final_answer
        FROM runs
        ORDER BY started_at ASC, run_id ASC
        """
    ).fetchall()


def _load_events(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            seq,
            event_type,
            label,
            content,
            tool_name,
            latency_ms,
            human_approval,
            metadata_json,
            created_at
        FROM events
        WHERE run_id = ?
        ORDER BY seq ASC, id ASC
        """,
        (run_id,),
    ).fetchall()


def _run_exists(conn: sqlite3.Connection, run_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM runs WHERE run_id = ? LIMIT 1", (run_id,)).fetchone()
    return row is not None


def merge_trace_db(dest_db: Path, source_db: Path) -> tuple[int, int]:
    if not source_db.is_file():
        raise FileNotFoundError(f"Source trace DB not found: {source_db}")

    dest_store = TraceStore(dest_db)
    imported_runs = 0
    skipped_runs = 0

    with sqlite3.connect(source_db) as src_conn, dest_store.connect() as dest_conn:
        src_conn.row_factory = sqlite3.Row
        dest_conn.row_factory = sqlite3.Row

        for run in _load_runs(src_conn):
            run_id = str(run["run_id"])
            if _run_exists(dest_conn, run_id):
                skipped_runs += 1
                continue

            dest_conn.execute(
                """
                INSERT INTO runs (
                    run_id,
                    task,
                    project_directory,
                    target_project,
                    model,
                    log_path,
                    started_at,
                    finished_at,
                    status,
                    final_answer
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run["task"],
                    run["project_directory"],
                    run["target_project"],
                    run["model"],
                    run["log_path"],
                    run["started_at"],
                    run["finished_at"],
                    run["status"],
                    run["final_answer"],
                ),
            )

            events = _load_events(src_conn, run_id)
            for event in events:
                dest_conn.execute(
                    """
                    INSERT INTO events (
                        run_id,
                        seq,
                        event_type,
                        label,
                        content,
                        tool_name,
                        latency_ms,
                        human_approval,
                        metadata_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        event["seq"],
                        event["event_type"],
                        event["label"],
                        event["content"],
                        event["tool_name"],
                        event["latency_ms"],
                        event["human_approval"],
                        event["metadata_json"],
                        event["created_at"],
                    ),
                )

            imported_runs += 1

        dest_conn.commit()

    return imported_runs, skipped_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge trace SQLite DBs from other projects into one destination DB."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / ".runs" / "traces.sqlite3",
        help="Destination trace DB path. Defaults to this repo's .runs/traces.sqlite3",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="Source trace DB paths or project directories containing .runs/traces.sqlite3",
    )
    args = parser.parse_args(argv)

    dest_db = args.dest.expanduser().resolve()
    total_imported = 0
    total_skipped = 0

    for raw_source in args.sources:
        source_db = _resolve_db_path(raw_source).resolve()
        if source_db == dest_db:
            print(f"skip {source_db} (same as destination)")
            continue

        imported, skipped = merge_trace_db(dest_db, source_db)
        total_imported += imported
        total_skipped += skipped
        print(f"{source_db}: imported={imported} skipped_existing={skipped}")

    print(f"done: imported={total_imported} skipped_existing={total_skipped} dest={dest_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
