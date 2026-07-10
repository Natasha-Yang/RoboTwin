---
name: sync-claude
description: Sync ALL portable Claude Code state for this repo across machines — both conversation transcripts and project memory — in one step. Use when the user says "sync claude", "sync everything", "save my claude state", "back this up", or just wants conversations + memory pushed/pulled together. Each store lives in its own separate PRIVATE git repo (RoboTwinConvos, RoboTwinMemory) nested under .claude/.
---

# Sync all Claude state (conversations + memory)

This repo keeps two portable Claude stores, each in a **separate private git repo**
nested under `.claude/` and gitignored by the public fork:

- **conversations** → `.claude/conversations/` (remote `RoboTwinConvos`)
- **memory** → `.claude/memory/` (remote `RoboTwinMemory`)

See `.claude/README.md` for the design. Run from a **login node** (network
required); compute nodes are offline.

## Default action — sync everything (both private stores + the config)

```bash
bash .claude/hooks/sync-conversations.sh sync   # private store: RoboTwinConvos
bash .claude/hooks/sync-memory.sh sync          # private store: RoboTwinMemory
bash .claude/hooks/sync-claude-config.sh        # .claude/** + CLAUDE.md -> main AND cluster
```

The two store syncs each do: export this machine's live state → commit → pull/merge
other machines' state → import merged set → **push**. Every call ends with a push.

The **config** sync propagates the tracked Claude *utility* files — everything under
`.claude/` (skills, hooks, `settings.json`, statusline) **plus `CLAUDE.md`** — to
**both** the `main` and `cluster` branches of the main (public) repo, then pushes
each. It commits **only** those config paths on each branch (via a throwaway git
worktree for whichever branch isn't checked out), so it never merges unrelated dev
work between branches and never touches your other uncommitted changes. This is what
keeps `main` from drifting behind `cluster` on the Claude tooling. Pass `--no-push`
to commit on both branches without pushing.

Report to the user, per store, what was committed, whether the pull brought anything
in, and that the push succeeded; and for the config, which branches got updated/pushed.

## Scope it down if asked

- Conversations only → run the `/sync-conversations` skill (or the first command).
- Memory only → run the `/sync-memory` skill (or the second command).
- Push-only / pull-only per store: pass `save` or `pull` instead of `sync`.
- Config only (propagate `.claude/**` + `CLAUDE.md` to both branches) → just the
  third command; add `--no-push` to stage the commits locally without pushing.

## Guidance

- If a script reports the nested folder isn't a git repo, the user is on a fresh
  clone — clone the relevant private repo into place first (see `.claude/README.md`).
- Never move either store into the main repo — it's a public fork.
- `gh`'s API can't see these private repos (fine-grained PAT scope), but git-over-SSH
  works, which is all the sync uses. If push fails, check SSH access to GitHub.
