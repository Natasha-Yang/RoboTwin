#!/usr/bin/env bash
# Sync Claude Code project memory between this machine's live store
# (~/.claude/projects/<path-hash>/memory/) and the git-tracked, gitignored copy in
# the repo (.claude/memory/), backed by a separate PRIVATE remote. Thin wrapper
# over sync-store.sh; see it and .claude/README.md for details.
#
#   sync-memory.sh [sync|save|pull|export|import]   (default: sync)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/sync-store.sh" memory "${1:-sync}"
