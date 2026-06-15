# Trace / Run Viewer

Every agent run still writes the readable `.runs/*.log` file, and now also writes structured events to:

```text
.runs/traces.sqlite3
```

List recent runs:

```powershell
.\.venv\Scripts\python.exe scripts\view_runs.py list
```

Show the latest run:

```powershell
.\.venv\Scripts\python.exe scripts\view_runs.py show
```

Show a specific run:

```powershell
.\.venv\Scripts\python.exe scripts\view_runs.py show <run_id>
```

The report includes:

- task, status, model, log path, start and finish time
- plan steps when planning was used
- Thought / Action / Observation events
- tool policy audit events with risk, approval, and blocked reason metadata
- tool result latency
- human approval events and whether they were accepted
- final answer
- evidence ledger
