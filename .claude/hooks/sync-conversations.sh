#!/usr/bin/env bash
# Sync Claude Code conversation transcripts between this machine's live store
# (~/.claude/projects/<path-hash>/) and the git-tracked copy in the repo
# (.claude/conversations/). This lets you commit conversations on one machine
# and resume them on any other clone, regardless of the repo's absolute path.
#
#   sync-conversations.sh export   local live store  ->  repo (.claude/conversations)
#   sync-conversations.sh import   repo              ->  local live store
#
# Wired to SessionEnd (export) and SessionStart (import) in .claude/settings.json.
# When invoked by a hook, the transcript_path from the hook's stdin JSON is used
# to locate the live store exactly; for manual runs the path hash is derived from
# the repo's absolute path the same way Claude Code does (/ and . -> -).
set -euo pipefail

DIRECTION="${1:-export}"

# Resolve repo root (.claude/hooks/ -> repo root). Prefer the env Claude sets.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
REPO_CONV="$REPO_DIR/.claude/conversations"
mkdir -p "$REPO_CONV"

# Try to read hook stdin JSON (non-blocking); use transcript_path if present.
STDIN_JSON=""
if [ ! -t 0 ]; then
  STDIN_JSON="$(cat || true)"
fi

LIVE_DIR=""
if [ -n "$STDIN_JSON" ]; then
  LIVE_DIR="$(printf '%s' "$STDIN_JSON" | python3 -c 'import sys,json,os
try:
    d=json.load(sys.stdin); p=d.get("transcript_path")
    print(os.path.dirname(p) if p else "")
except Exception:
    print("")' 2>/dev/null || true)"
fi

# Fallback: derive the live store dir from the repo path the way Claude Code encodes it.
if [ -z "$LIVE_DIR" ]; then
  HASH="$(printf '%s' "$REPO_DIR" | sed -e 's/[/.]/-/g')"
  LIVE_DIR="$HOME/.claude/projects/$HASH"
fi

copy() { # copy *.jsonl from $1 to $2, only when newer or missing
  local src="$1" dst="$2"
  [ -d "$src" ] || return 0
  mkdir -p "$dst"
  shopt -s nullglob
  local f had=0
  for f in "$src"/*.jsonl; do
    had=1
    cp -u "$f" "$dst"/ 2>/dev/null || cp "$f" "$dst"/
  done
  shopt -u nullglob
  return 0
}

case "$DIRECTION" in
  export) copy "$LIVE_DIR" "$REPO_CONV" ;;
  import) copy "$REPO_CONV" "$LIVE_DIR" ;;

  # save: export live -> nested repo, then commit + push the PRIVATE repo.
  # Requires network (run from a login node). No-op commit if nothing changed.
  save)
    copy "$LIVE_DIR" "$REPO_CONV"
    if [ ! -d "$REPO_CONV/.git" ]; then
      echo "error: $REPO_CONV is not a git repo. Clone the private conversations" >&2
      echo "       repo there first (see .claude/README.md)." >&2
      exit 1
    fi
    git -C "$REPO_CONV" add -A
    if git -C "$REPO_CONV" diff --cached --quiet; then
      echo "conversations: nothing new to save"
    else
      git -C "$REPO_CONV" commit -q -m "sync conversations: $(date -u +%Y-%m-%dT%H:%MZ) from $(hostname -s)"
      echo "conversations: committed."
    fi
    if git -C "$REPO_CONV" remote get-url origin >/dev/null 2>&1; then
      git -C "$REPO_CONV" push -q origin HEAD && echo "conversations: pushed to origin."
    else
      echo "conversations: no 'origin' remote set — commit saved locally only." >&2
    fi
    ;;

  # pull: fetch the latest from the private repo, then import into the live store.
  pull)
    if [ -d "$REPO_CONV/.git" ] && git -C "$REPO_CONV" remote get-url origin >/dev/null 2>&1; then
      git -C "$REPO_CONV" pull -q --ff-only origin HEAD 2>/dev/null \
        || git -C "$REPO_CONV" pull -q origin "$(git -C "$REPO_CONV" branch --show-current)" \
        || echo "conversations: pull skipped (offline or no upstream)"
    fi
    copy "$REPO_CONV" "$LIVE_DIR"
    echo "conversations: imported into live store — use 'claude --resume'."
    ;;

  *) echo "usage: sync-conversations.sh [export|import|save|pull]" >&2; exit 2 ;;
esac

exit 0
