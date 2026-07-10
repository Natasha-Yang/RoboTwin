#!/usr/bin/env bash
# Propagate this repo's tracked Claude *utility* files to BOTH the main and
# cluster branches, then push each. This keeps the Claude Code tooling identical
# across branches (e.g. `main` was missing every `.claude/` change developed on
# `cluster`) WITHOUT merging unrelated dev work between the branches — only the
# config paths below are touched on each branch.
#
# What is propagated: the tracked files under `.claude/` PLUS `CLAUDE.md`. The
# private stores (`.claude/conversations/`, `.claude/memory/`) are gitignored and
# handled by their own per-store sync (sync-conversations.sh / sync-memory.sh) —
# they must NEVER enter the public fork, and `git ls-files` already excludes them.
#
#   sync-claude-config.sh            propagate + push both branches (default)
#   sync-claude-config.sh --no-push  commit locally on both branches, don't push
#
# Source of truth = the CURRENT branch's working-tree copy of the config paths
# (so uncommitted config edits are captured). Run from a login node (push needs
# network). Requires a clean-enough state: only the config paths are committed;
# your other uncommitted changes are left untouched.
set -euo pipefail

PUSH=1
[ "${1:-}" = "--no-push" ] && PUSH=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_DIR"

BRANCHES=(main cluster)
CURRENT="$(git branch --show-current)"

# Config files = tracked PLUS not-yet-tracked (so this script bootstraps itself on
# its first run). `--exclude-standard` drops gitignored paths, i.e. the private
# stores `.claude/conversations/` and `.claude/memory/` — they must never enter the
# public fork and are handled by their own per-store sync.
mapfile -t FILES < <(git ls-files --cached --others --exclude-standard -- CLAUDE.md .claude)
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "config: no tracked CLAUDE.md/.claude files found — nothing to propagate." >&2
  exit 1
fi

STAMP="$(date -u +%Y-%m-%dT%H:%MZ)"
HOST="$(hostname -s)"
MSG="sync claude config: $STAMP from $HOST"

commit_in_place() { # commit ONLY the config paths on the current branch
  git add -- "${FILES[@]}"
  if git diff --cached --quiet -- "${FILES[@]}"; then
    echo "config[$CURRENT]: already up to date."
  else
    git commit -q -o -m "$MSG" -- "${FILES[@]}"
    echo "config[$CURRENT]: committed config changes."
  fi
}

propagate_via_worktree() { # make branch $1 match the current config files
  local br="$1" wt
  wt="$(mktemp -d "${TMPDIR:-/tmp}/claude-cfg-${br}-XXXXXX")"
  # Detached worktree on the target branch's tip; --force tolerates a reused path.
  if ! git worktree add -q --force "$wt" "$br" 2>/dev/null; then
    echo "config[$br]: could not create worktree (branch missing?) — skipped." >&2
    rm -rf "$wt"; return 0
  fi
  # Overlay the current branch's config files onto the target worktree.
  local f
  for f in "${FILES[@]}"; do
    mkdir -p "$wt/$(dirname "$f")"
    cp "$REPO_DIR/$f" "$wt/$f"
  done
  git -C "$wt" add -- "${FILES[@]}"
  if git -C "$wt" diff --cached --quiet -- "${FILES[@]}"; then
    echo "config[$br]: already up to date."
  else
    git -C "$wt" commit -q -m "$MSG" -- "${FILES[@]}"
    echo "config[$br]: committed config changes."
    if [ "$PUSH" -eq 1 ]; then
      git -C "$wt" push -q origin "HEAD:$br" \
        && echo "config[$br]: pushed to origin." \
        || echo "config[$br]: push failed (offline or remote diverged)." >&2
    fi
  fi
  git worktree remove -q --force "$wt" 2>/dev/null || rm -rf "$wt"
  git worktree prune -q 2>/dev/null || true
}

for br in "${BRANCHES[@]}"; do
  if [ "$br" = "$CURRENT" ]; then
    commit_in_place
    if [ "$PUSH" -eq 1 ]; then
      git push -q origin "HEAD:$br" \
        && echo "config[$br]: pushed to origin." \
        || echo "config[$br]: push failed (offline or remote diverged)." >&2
    fi
  else
    propagate_via_worktree "$br"
  fi
done

echo "config: propagation complete (${BRANCHES[*]})."
exit 0
