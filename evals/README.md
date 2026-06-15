# Agent Eval Suite

Deterministic trace-grading suite for the local ReAct agent workflow.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_evals.py
```

Verbose per-case output:

```powershell
.\.venv\Scripts\python.exe scripts\run_evals.py --verbose
```

JSON output:

```powershell
.\.venv\Scripts\python.exe scripts\run_evals.py --json
```

The suite reads `evals/cases.json` and grades 50 standard workflow tasks:

- must call `read_file` for evidence-backed file tasks
- repeated `read_file` calls with the same target and budget should be blocked
- skill-oriented tasks should call `load_skill` before file work
- terminal commands, including dangerous commands, must trigger confirmation
- MCP `find_symbol` results must be reflected in the final summary
- tool errors should be observed and recovered from cleanly

Default text report:

```text
total=50
pass=50
pass_rate=100%
avg_steps=2.8
tool_error_rate=5.9%
repeated_read_blocked=8
```

