#!/usr/bin/env bash
#
# guard-edits.sh — unified PreToolUse guard (WSL/bash port of guard-edits.ps1).
#
# Reads the Claude Code PreToolUse payload on stdin and dispatches on tool_name:
#
#   Edit/Write/MultiEdit/NotebookEdit:
#     denies any edit whose target is NOT inside <project>/src/ or
#     <project>/paper/, or exactly <project>/CLAUDE.md.
#
#   Bash:
#     git guard — denies git commit/push/merge/rebase/cherry-pick/am on
#       protected branches, and denies hook-bypass tricks on any branch
#       (--no-verify, -n on commit-family commands, core.hooksPath overrides).
#     gh guard  — GH_TOKEN is scoped to issues; deny gh subcommands outside
#       issue create/comment/list/view (no close/edit/delete, no pr/repo/api).
#
# Wired up in .claude/settings.json on the Edit|Write|MultiEdit|NotebookEdit|Bash
# matcher. To change the allowed paths, edit `allowed_dirs`/`claude_md`; to
# change the protected branches, edit PROTECTED_BRANCHES in the embedded Python.
#
# Fails CLOSED (deny) on an unparseable/missing payload, a missing target path,
# a missing python3, or an undeterminable git branch — an unreadable input must
# not slip past the sandbox.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

deny() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}\n' "$1"
  exit 0
}

command -v python3 >/dev/null 2>&1 || deny "Sandbox guard: python3 not found; blocking tool call as a precaution."

ROOT="$ROOT" python3 -c '
import sys, os, re, json, subprocess

# === EDIT THESE ==============================================================
PROTECTED_BRANCHES = {"siamese_net", "main"}          # workhorse branches
GH_ALLOWED = {"create", "comment", "list", "view", "status"}  # gh issue subcommands
# =============================================================================

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
    deny("Sandbox guard could not parse the tool payload; blocking as a precaution.")

root = os.environ["ROOT"]
tool = payload.get("tool_name") or ""
ti = payload.get("tool_input") or {}

# ---------------------------------------------------------------------------
# Bash branch: git guard + gh guard
# ---------------------------------------------------------------------------
if tool == "Bash":
    cmd = ti.get("command") or ""

    # --- git guard ---------------------------------------------------------
    if re.search(r"\bgit\b", cmd):
        write_ops = re.search(
            r"\bgit\b[^|;&\n]*\b(commit|push|merge|rebase|cherry-pick|am)\b", cmd)

        if write_ops:
            # Block hook-bypass flags regardless of branch.
            if re.search(r"--no-verify\b", cmd) or re.search(r"(^|\s)-n(\s|$)", cmd):
                deny("Git guard: --no-verify / -n is never allowed. "
                     "Fix the underlying issue instead of bypassing hooks.")

            # Block hooksPath overrides (another way to dodge pre-commit hooks).
            if re.search(r"core\.hooksPath", cmd, re.IGNORECASE):
                deny("Git guard: overriding core.hooksPath is not allowed.")

            # Determine the current branch; fail closed if we cannot.
            # (branch --show-current works on unborn branches, unlike
            # rev-parse --abbrev-ref HEAD which prints the literal "HEAD";
            # it prints nothing on a detached HEAD, which we also block.)
            try:
                res = subprocess.run(
                    ["git", "-C", root, "branch", "--show-current"],
                    capture_output=True, text=True, timeout=10)
                branch = res.stdout.strip() if res.returncode == 0 else ""
            except Exception:
                branch = ""
            if not branch or branch == "HEAD":
                deny("Git guard: could not determine the current branch "
                     "(detached HEAD or repo error); blocking git write "
                     "operation as a precaution.")

            if branch in PROTECTED_BRANCHES:
                deny("Git guard: no commits/pushes/merges on protected branch "
                     f"{branch!r}. Do not retry or work around this: summarize "
                     "your changes and ask the user to review and commit.")

    # --- gh guard ----------------------------------------------------------
    # GH_TOKEN is issue-scoped by design; enforce the same scope at the
    # command level so misuse fails loudly instead of via API errors.
    # Match gh only in COMMAND position (start of line/segment, after a
    # shell separator, or after VAR=val prefixes) — not as a bare argument
    # to which/find/apt etc.
    GH_CMD = re.compile(
        r"(?:^|[;&|(\n]|\$\()\s*(?:[A-Za-z_][A-Za-z_0-9]*=\S*\s+)*gh\s+(\S+)(?:\s+(\S+))?")
    m = GH_CMD.search(cmd)
    if m:
        sub = m.group(1) or ""
        action = m.group(2) or ""
        if sub != "issue":
            deny(f"gh guard: only \"gh issue ...\" is allowed (got \"gh {sub}\"). "
                 "The token is issue-scoped; do not use gh for anything else.")
        if action not in GH_ALLOWED:
            deny(f"gh guard: \"gh issue {action}\" is not allowed. "
                 "Permitted: create, comment, list, view, status. "
                 "Never close, edit, or delete issues — the user closes them via commits.")

    raise SystemExit(0)  # Bash command is fine -> proceed

# ---------------------------------------------------------------------------
# Edit/Write/MultiEdit/NotebookEdit branch: path allowlist
# ---------------------------------------------------------------------------
# Edit/Write/MultiEdit use file_path; NotebookEdit uses notebook_path.
fp = ti.get("file_path") or ti.get("notebook_path")
if not fp:
    deny("Sandbox guard found no target path on the edit; blocking as a precaution.")

full = os.path.realpath(fp if os.path.isabs(fp) else os.path.join(root, fp))

allowed_dirs = [os.path.realpath(os.path.join(root, "src")),
                os.path.realpath(os.path.join(root, "paper")),
                os.path.realpath(os.path.join(root, ".claude", "plans")),
                os.path.realpath(os.path.expanduser("~/.claude/plans"))]
claude_md = os.path.realpath(os.path.join(root, "CLAUDE.md"))

# /mnt/c is case-insensitive; compare case-folded and on directory boundaries.
fl = full.lower()
ok = (fl == claude_md.lower()) or any(
    fl == d.lower() or fl.startswith(d.lower() + os.sep) for d in allowed_dirs)
if not ok:
    deny("Sandboxed: edits are allowed only under src/, paper/, or to CLAUDE.md. Blocked: " + str(fp))
'