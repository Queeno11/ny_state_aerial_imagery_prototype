#!/usr/bin/env bash
#
# guard-edits.sh — PreToolUse allowlist guard (WSL/bash port of guard-edits.ps1).
#
# Reads the Claude Code PreToolUse payload on stdin and denies any
# Edit/Write/MultiEdit/NotebookEdit whose target is NOT inside <project>/src/
# or <project>/paper/, or exactly <project>/CLAUDE.md.
#
# Wired up in .claude/settings.json on the Edit|Write|MultiEdit|NotebookEdit
# matcher. To change the allowed paths, edit the `allowed`/`claude_md` lines in
# the embedded Python below. To disable the sandbox, remove the PreToolUse block
# from .claude/settings.json (or use /hooks).
#
# Fails CLOSED (deny) on an unparseable/missing payload, a missing target path,
# or a missing python3 — an unreadable input must not slip past the sandbox.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

deny() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}\n' "$1"
  exit 0
}

command -v python3 >/dev/null 2>&1 || deny "Sandbox guard: python3 not found; blocking edit as a precaution."

ROOT="$ROOT" python3 -c '
import sys, os, json

def deny(reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason}}))
    raise SystemExit(0)

raw = sys.stdin.read()
if not raw.strip():
    raise SystemExit(0)  # nothing to inspect -> let normal flow proceed
try:
    payload = json.loads(raw)
except Exception:
    deny("Sandbox guard could not parse the tool payload; blocking edit as a precaution.")

ti = payload.get("tool_input") or {}
# Edit/Write/MultiEdit use file_path; NotebookEdit uses notebook_path.
fp = ti.get("file_path") or ti.get("notebook_path")
if not fp:
    deny("Sandbox guard found no target path on the edit; blocking as a precaution.")

root = os.environ["ROOT"]
full = os.path.realpath(fp if os.path.isabs(fp) else os.path.join(root, fp))

allowed_dirs = [os.path.realpath(os.path.join(root, "src")),
                os.path.realpath(os.path.join(root, "paper"))]
claude_md = os.path.realpath(os.path.join(root, "CLAUDE.md"))

# /mnt/c is case-insensitive; compare case-folded and on directory boundaries.
fl = full.lower()
ok = (fl == claude_md.lower()) or any(
    fl == d.lower() or fl.startswith(d.lower() + os.sep) for d in allowed_dirs)
if not ok:
    deny("Sandboxed: edits are allowed only under src/, paper/, or to CLAUDE.md. Blocked: " + str(fp))
'
