#!/usr/bin/env bash
# Generic sync engine for a Claude "store" that lives under this machine's
# per-project dir (~/.claude/projects/<path-hash>/) and is mirrored to a
# git-tracked, gitignored folder in the repo backed by a SEPARATE PRIVATE remote.
# This decouples portable Claude state from the public fork's history.
#
#   sync-store.sh <store> <action>
#     <store>  = conversations | memory
#     <action> = sync   bidirectional: export + commit + pull/merge + import + PUSH (default)
#                save   push-only:     export + commit + push
#                pull   pull-only:     fetch + import into the live store
#                export local live store -> repo folder   (network-free; used by hooks)
#                import repo folder      -> local live store (network-free; used by hooks)
#
# The live store dir is located from the hook stdin JSON's transcript_path when
# present; otherwise the path hash is derived from the repo's absolute path the
# same way Claude Code encodes it (/ and . -> -).
set -euo pipefail

STORE="${1:?usage: sync-store.sh <conversations|memory> <action>}"
ACTION="${2:-sync}"

# Resolve repo root (.claude/hooks/ -> repo root). Prefer the env Claude sets.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# Try to read hook stdin JSON (non-blocking); use transcript_path if present.
STDIN_JSON=""
if [ ! -t 0 ]; then
  STDIN_JSON="$(cat || true)"
fi
BASE=""
if [ -n "$STDIN_JSON" ]; then
  BASE="$(printf '%s' "$STDIN_JSON" | python3 -c 'import sys,json,os
try:
    d=json.load(sys.stdin); p=d.get("transcript_path")
    print(os.path.dirname(p) if p else "")
except Exception:
    print("")' 2>/dev/null || true)"
fi
# Fallback: derive the per-project dir from the repo path the way Claude Code encodes it.
if [ -z "$BASE" ]; then
  HASH="$(printf '%s' "$REPO_DIR" | sed -e 's/[/.]/-/g')"
  BASE="$HOME/.claude/projects/$HASH"
fi

# ---- per-store parameters ----
case "$STORE" in
  conversations)
    LIVE_DIR="$BASE"
    GLOB="*.jsonl"
    REPO_STORE="$REPO_DIR/.claude/conversations"
    IMPORT_HINT="use 'claude --resume' to pick one up." ;;
  memory)
    LIVE_DIR="$BASE/memory"
    GLOB="*.md"
    REPO_STORE="$REPO_DIR/.claude/memory"
    IMPORT_HINT="memories are live for new sessions in this project." ;;
  *)
    echo "error: unknown store '$STORE' (want: conversations | memory)" >&2; exit 2 ;;
esac
mkdir -p "$REPO_STORE"

copy() { # copy $GLOB from $1 to $2, only when newer or missing
  local src="$1" dst="$2" f
  [ -d "$src" ] || return 0
  mkdir -p "$dst"
  shopt -s nullglob
  for f in "$src"/$GLOB; do
    cp -u "$f" "$dst"/ 2>/dev/null || cp "$f" "$dst"/
  done
  shopt -u nullglob
  return 0
}

require_git_repo() {
  if [ ! -d "$REPO_STORE/.git" ]; then
    echo "error: $REPO_STORE is not a git repo. Clone the private $STORE repo" >&2
    echo "       there first (see .claude/README.md)." >&2
    exit 1
  fi
}
has_origin() { git -C "$REPO_STORE" remote get-url origin >/dev/null 2>&1; }

git_commit_local() { # stage + commit whatever is in the nested repo; no-op if clean
  git -C "$REPO_STORE" add -A
  if git -C "$REPO_STORE" diff --cached --quiet; then
    echo "$STORE: nothing new to commit"
  else
    git -C "$REPO_STORE" commit -q -m "sync $STORE: $(date -u +%Y-%m-%dT%H:%MZ) from $(hostname -s)"
    echo "$STORE: committed local changes."
  fi
}
git_pull_remote() { # merge remote into local; tolerant of offline / first push
  has_origin || { echo "$STORE: no 'origin' remote — skipping pull." >&2; return 0; }
  local br; br="$(git -C "$REPO_STORE" branch --show-current)"
  git -C "$REPO_STORE" pull -q --no-edit origin "$br" 2>/dev/null \
    && echo "$STORE: pulled remote changes." \
    || echo "$STORE: pull skipped (offline, no upstream yet, or nothing to pull)."
}
git_push_remote() {
  has_origin || { echo "$STORE: no 'origin' remote — commit saved locally only." >&2; return 0; }
  git -C "$REPO_STORE" push -q origin HEAD \
    && echo "$STORE: pushed to origin." \
    || echo "$STORE: push failed (offline, or the remote repo doesn't exist yet)." >&2
}

case "$ACTION" in
  export) copy "$LIVE_DIR" "$REPO_STORE" ;;
  import) copy "$REPO_STORE" "$LIVE_DIR" ;;
  sync)
    require_git_repo
    copy "$LIVE_DIR" "$REPO_STORE"   # capture this machine's live state
    git_commit_local
    git_pull_remote                  # merge in other machines' state
    copy "$REPO_STORE" "$LIVE_DIR"   # push merged set back into the live store
    git_push_remote                  # always push
    echo "$STORE: sync complete (pulled + pushed)." ;;
  save)
    require_git_repo
    copy "$LIVE_DIR" "$REPO_STORE"
    git_commit_local
    git_push_remote ;;
  pull)
    require_git_repo
    git_pull_remote
    copy "$REPO_STORE" "$LIVE_DIR"
    echo "$STORE: imported into live store — $IMPORT_HINT" ;;
  *)
    echo "usage: sync-store.sh <conversations|memory> [sync|save|pull|export|import]" >&2
    exit 2 ;;
esac

exit 0
