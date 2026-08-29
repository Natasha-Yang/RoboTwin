#!/usr/bin/env bash
# Sync Claude Code conversation transcripts between this machine's live store
# (~/.claude/projects/<path-hash>/) and the git-tracked, gitignored copy in the
# repo (.claude/conversations/), backed by a separate PRIVATE remote. Thin wrapper
# over sync-store.sh; see it and .claude/README.md for details.
#
#   sync-conversations.sh [sync|save|pull|export|import]   (default: sync)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/sync-store.sh" conversations "${1:-sync}"
