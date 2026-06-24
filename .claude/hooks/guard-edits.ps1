<#
    guard-edits.ps1 — PreToolUse allowlist guard.

    Receives the Claude Code PreToolUse payload as JSON on stdin and denies any
    Edit/Write/MultiEdit/NotebookEdit whose target is NOT inside <project>/src/
    or exactly <project>/CLAUDE.md.

    Wired up in .claude/settings.json on the Edit|Write|MultiEdit|NotebookEdit
    matcher. To change the allowed paths, edit $allowed below. To disable the
    sandbox, remove the PreToolUse block from .claude/settings.json (or use /hooks).

    Exit 0 + no output  -> stay silent, let normal permission flow proceed (allow).
    Exit 0 + deny JSON   -> hard-block the edit with a reason.
#>

$ErrorActionPreference = 'Stop'

function Deny([string]$reason) {
    @{ hookSpecificOutput = @{
        hookEventName            = 'PreToolUse'
        permissionDecision       = 'deny'
        permissionDecisionReason = $reason
    } } | ConvertTo-Json -Compress -Depth 5
    exit 0
}

try {
    $raw = [Console]::In.ReadToEnd()
    if ([string]::IsNullOrWhiteSpace($raw)) { exit 0 }
    $payload = $raw | ConvertFrom-Json
} catch {
    # Fail closed: an unreadable payload must not slip past the sandbox.
    Deny "Sandbox guard could not parse the tool payload; blocking edit as a precaution."
}

# Edit/Write/MultiEdit use file_path; NotebookEdit uses notebook_path.
$fp = $payload.tool_input.file_path
if ([string]::IsNullOrWhiteSpace($fp)) { $fp = $payload.tool_input.notebook_path }
if ([string]::IsNullOrWhiteSpace($fp)) {
    Deny "Sandbox guard found no target path on the edit; blocking as a precaution."
}

# Project root = two levels up from this script (.claude/hooks -> .claude -> root).
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$full     = [System.IO.Path]::GetFullPath($fp)
$sep      = [System.IO.Path]::DirectorySeparatorChar
$srcDir   = [System.IO.Path]::GetFullPath((Join-Path $root 'src')) + $sep
$claudeMd = [System.IO.Path]::GetFullPath((Join-Path $root 'CLAUDE.md'))

$ic = [System.StringComparison]::OrdinalIgnoreCase
$allowed = $full.StartsWith($srcDir, $ic) -or $full.Equals($claudeMd, $ic)

if ($allowed) { exit 0 }

Deny "Sandboxed: edits are allowed only under src/ or to CLAUDE.md. Blocked: $fp"
