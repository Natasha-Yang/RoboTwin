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

## Default action — sync both (bidirectional: pull AND push)

```bash
bash .claude/hooks/sync-conversations.sh sync
bash .claude/hooks/sync-memory.sh sync
```

Each does: export this machine's live state → commit → pull/merge other machines'
state → import merged set → **push**. Every call ends with a push. Report to the
user, per store, what was committed, whether the pull brought anything in, and that
the push succeeded.

## Scope it down if asked

- Conversations only → run the `/sync-conversations` skill (or the first command).
- Memory only → run the `/sync-memory` skill (or the second command).
- Push-only / pull-only per store: pass `save` or `pull` instead of `sync`.

## Guidance

- If a script reports the nested folder isn't a git repo, the user is on a fresh
  clone — clone the relevant private repo into place first (see `.claude/README.md`).
- Never move either store into the main repo — it's a public fork.
- `gh`'s API can't see these private repos (fine-grained PAT scope), but git-over-SSH
  works, which is all the sync uses. If push fails, check SSH access to GitHub.
