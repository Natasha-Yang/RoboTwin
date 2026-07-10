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

require_git_repo() {
  if [ ! -d "$REPO_CONV/.git" ]; then
    echo "error: $REPO_CONV is not a git repo. Clone the private conversations" >&2
    echo "       repo there first (see .claude/README.md)." >&2
    exit 1
  fi
}
has_origin() { git -C "$REPO_CONV" remote get-url origin >/dev/null 2>&1; }

git_commit_local() { # stage + commit whatever is in the nested repo; no-op if clean
  git -C "$REPO_CONV" add -A
  if git -C "$REPO_CONV" diff --cached --quiet; then
    echo "conversations: nothing new to commit"
  else
    git -C "$REPO_CONV" commit -q -m "sync conversations: $(date -u +%Y-%m-%dT%H:%MZ) from $(hostname -s)"
    echo "conversations: committed local changes."
  fi
}
git_pull_remote() { # merge remote into local; tolerant of offline / first push
  has_origin || { echo "conversations: no 'origin' remote — skipping pull." >&2; return 0; }
  local br; br="$(git -C "$REPO_CONV" branch --show-current)"
  git -C "$REPO_CONV" pull -q --no-edit origin "$br" 2>/dev/null \
    && echo "conversations: pulled remote changes." \
    || echo "conversations: pull skipped (offline, no upstream yet, or nothing to pull)."
}
git_push_remote() {
  has_origin || { echo "conversations: no 'origin' remote — commit saved locally only." >&2; return 0; }
  git -C "$REPO_CONV" push -q origin HEAD \
    && echo "conversations: pushed to origin." \
    || echo "conversations: push failed (offline, or the remote repo doesn't exist yet)." >&2
}

case "$DIRECTION" in
  # low-level file copies (network-free); used by the SessionStart/SessionEnd hooks.
  export) copy "$LIVE_DIR" "$REPO_CONV" ;;
  import) copy "$REPO_CONV" "$LIVE_DIR" ;;

  # sync (DEFAULT for the /sync-conversations skill): fully bidirectional.
  # export local -> commit -> pull+merge remote -> import merged set -> PUSH.
  # Every call ends with a push. Run from a login node (needs network).
  sync)
    require_git_repo
    copy "$LIVE_DIR" "$REPO_CONV"   # capture this machine's live transcripts
    git_commit_local                # commit them
    git_pull_remote                 # merge in other machines' transcripts
    copy "$REPO_CONV" "$LIVE_DIR"   # push merged set back into the live store
    git_push_remote                 # always push
    echo "conversations: sync complete (pulled + pushed)."
    ;;

  # save: push-only (export local -> commit -> push). Skips the pull step.
  save)
    require_git_repo
    copy "$LIVE_DIR" "$REPO_CONV"
    git_commit_local
    git_push_remote
    ;;

  # pull: pull-only (fetch remote -> import to live store).
  pull)
    require_git_repo
    git_pull_remote
    copy "$REPO_CONV" "$LIVE_DIR"
    echo "conversations: imported into live store — use 'claude --resume'."
    ;;

  *) echo "usage: sync-conversations.sh [sync|save|pull|export|import]" >&2; exit 2 ;;
esac

exit 0
