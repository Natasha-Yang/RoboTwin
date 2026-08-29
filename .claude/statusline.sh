#!/usr/bin/env bash
# Claude Code status line for the RoboTwin repo.
# Passes the status-line JSON payload (stdin) straight to statusline.py.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/statusline.py"
