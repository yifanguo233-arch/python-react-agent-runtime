from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_traces import TraceStore, format_run_list, format_run_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="View recorded agent traces.")
    parser.add_argument("--db", type=Path, default=ROOT / ".runs" / "traces.sqlite3")
    parser.add_argument("--project", help="Filter runs by target project (or runtime project if target is empty).")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List recent runs.")
    list_parser.add_argument("--limit", type=int, default=20)

    show_parser = subparsers.add_parser("show", help="Show one run trace.")
    show_parser.add_argument("run_id", nargs="?", help="Run id. Defaults to the latest run.")

    args = parser.parse_args(argv)
    command = args.command or "list"
    store = TraceStore(args.db)

    if command == "list":
        print(format_run_list(store.list_runs(limit=args.limit, project=args.project)))
        return 0

    if command == "show":
        run_id = args.run_id or store.latest_run_id(project=args.project)
        if not run_id:
            print("No runs recorded yet.")
            return 1
        print(format_run_report(store, run_id))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
