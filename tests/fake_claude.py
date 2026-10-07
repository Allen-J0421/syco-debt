#!/usr/bin/env python3
"""Stand-in for the Claude Code CLI in tests. Accepts the same flags the pipeline passes and
emits stream-json. Behaviour per turn comes from FAKE_CLAUDE_PLAN (comma-separated):

  edit        append a line to the target file
  stop        change nothing, decline in words
  claimedit   decline in words but edit anyway
  selfcommit  edit and commit the change itself
  ask         change nothing and ask a clarifying question (answers with an edit when replied to)
  limit       report a provider session limit (only the first time across all sessions)
  fail        exit non-zero
"""
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

args = sys.argv[1:]
prompt = args[args.index("-p") + 1]
sid = args[args.index("--resume") + 1] if "--resume" in args else str(uuid.uuid4())
state_dir = Path(os.environ["FAKE_CLAUDE_STATE"])
state_dir.mkdir(parents=True, exist_ok=True)
state_file = state_dir / f"{sid}.json"
state = json.loads(state_file.read_text()) if state_file.exists() else {"calls": 0, "target": "", "asked": False}

if not state["target"]:
    m = re.search(r"in `([^`]+)`", prompt)
    state["target"] = m.group(1) if m else ""

plan = os.environ.get("FAKE_CLAUDE_PLAN", "edit").split(",")
answering = state.get("asked", False)
if answering:
    action = "edit"
    state["asked"] = False
else:
    action = plan[min(state["calls"], len(plan) - 1)]
    state["calls"] += 1

time.sleep(float(os.environ.get("FAKE_CLAUDE_SLEEP", "0.02")))


def emit(obj):
    print(json.dumps(obj), flush=True)


def edit():
    target = Path(state["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as f:
        f.write(f"// edit {state['calls']} by {sid[:6]}\n")


emit({"type": "system", "subtype": "init", "session_id": sid})
text = "Refactored."
if action == "limit":
    marker = state_dir / "limit.fired"
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        state["calls"] -= 1
        state_file.write_text(json.dumps(state))
        emit({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
              "result": "You've hit your session limit", "session_id": sid})
        sys.exit(0)
    except FileExistsError:
        action = "edit"
if action == "fail":
    state_file.write_text(json.dumps(state))
    sys.exit(7)
if action == "edit":
    edit()
elif action == "stop":
    text = "I'm not going to refactor this again. The code is already clean."
elif action == "claimedit":
    edit()
    text = "I'm not going to refactor this again, but I made one small change."
elif action == "selfcommit":
    edit()
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "agent's own commit"], check=True)
elif action == "ask":
    state["asked"] = True
    text = "Could you tell me which part matters most to you?"

state_file.write_text(json.dumps(state))
emit({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}, "session_id": sid})
emit({"type": "result", "subtype": "success", "is_error": False, "result": text, "session_id": sid,
      "total_cost_usd": 0.01, "duration_api_ms": 20, "num_turns": 1,
      "usage": {"input_tokens": 10, "output_tokens": 5}})
