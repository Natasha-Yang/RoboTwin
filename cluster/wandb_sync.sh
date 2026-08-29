#!/bin/bash
# ---------------------------------------------------------------------------
# Sync offline wandb runs to wandb.ai. RUN FROM A LOGIN NODE (needs internet).
#
# pi0.5 fine-tuning runs on compute nodes with no internet, so wandb is set to
# offline mode (WANDB_MODE=offline in cluster/finetune_pi05.sh). That writes each
# run to policy/pi05/wandb/{offline-,}run-<ts>-<id>/ but never uploads it. This
# script pushes those local runs up so they show in the wandb web UI.
#
# Usage:
#   bash cluster/wandb_sync.sh                 # sync ALL local runs
#   bash cluster/wandb_sync.sh latest          # sync only the most recent run
#   bash cluster/wandb_sync.sh <run_dir> ...   # sync specific run dir(s)
#
# Safe to run mid-training: syncing an in-progress offline run uploads whatever
# has been logged so far; re-run later to push newer steps.
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve repo root (this script lives in <root>/cluster/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WANDB_DIR="$ROOT/policy/pi05/wandb"
WANDB_BIN="$ROOT/policy/pi05/.venv/bin/wandb"

[ -x "$WANDB_BIN" ] || { echo "ERROR: wandb not found at $WANDB_BIN (run 'uv sync' in policy/pi05 first)" >&2; exit 1; }
[ -d "$WANDB_DIR" ] || { echo "ERROR: no wandb dir at $WANDB_DIR (nothing to sync)" >&2; exit 1; }

# Guard against the Alliance CVMFS python leak (dummy opencv wheel, pip config).
unset PYTHONPATH PIP_CONFIG_FILE 2>/dev/null || true

cd "$ROOT/policy/pi05"

if [ "$#" -eq 0 ]; then
    echo "== syncing ALL runs in $WANDB_DIR =="
    "$WANDB_BIN" sync --sync-all
elif [ "$1" = "latest" ]; then
    target="$(readlink -f "$WANDB_DIR/latest-run" 2>/dev/null || true)"
    [ -n "$target" ] && [ -d "$target" ] || { echo "ERROR: no latest-run under $WANDB_DIR" >&2; exit 1; }
    echo "== syncing latest run: $target =="
    "$WANDB_BIN" sync "$target"
else
    echo "== syncing $# specified run(s) =="
    "$WANDB_BIN" sync "$@"
fi
