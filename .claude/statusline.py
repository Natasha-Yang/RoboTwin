#!/usr/bin/env python3
"""Claude Code status line renderer for the RoboTwin repo.

Reads the status-line JSON payload on stdin and prints:
    <mode> · <model> · ctx <used>/<window> (<pct>)
Docs: https://docs.claude.com/en/docs/claude-code/statusline
"""
import sys, json, os

try:
    data = json.load(sys.stdin)
except Exception:
    data = {}

model = (data.get("model") or {}).get("display_name") \
    or (data.get("model") or {}).get("id") or "?"

# Operating/permission mode (default | acceptEdits | plan | bypassPermissions)
mode = data.get("permissionMode") or data.get("mode") or ""
mode_label = {
    "acceptEdits": "auto",
    "bypassPermissions": "bypass",
    "plan": "plan",
    "default": "normal",
}.get(mode, mode or "")

# ---- context usage: derive from the transcript's most recent usage record ----
WINDOW = 200_000  # standard context window (1M not enabled on this account)
used = 0
tpath = data.get("transcript_path")
if tpath and os.path.exists(tpath):
    try:
        with open(tpath, "rb") as f:
            lines = f.read().splitlines()
        for raw in reversed(lines):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            usage = ((obj.get("message") or {}).get("usage")) or obj.get("usage")
            if usage:
                used = (
                    (usage.get("input_tokens") or 0)
                    + (usage.get("cache_read_input_tokens") or 0)
                    + (usage.get("cache_creation_input_tokens") or 0)
                    + (usage.get("output_tokens") or 0)
                )
                break
    except Exception:
        used = 0

pct = int(round(100 * used / WINDOW)) if WINDOW else 0


def human(n):
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


parts = []
if mode_label:
    parts.append(mode_label)
parts.append(model)
parts.append(f"ctx {human(used)}/{human(WINDOW)} ({pct}%)")
print(" · ".join(parts))
