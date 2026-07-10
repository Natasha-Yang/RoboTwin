---
name: sync-memory
description: Save, back up, or sync Claude Code project MEMORY for this repo across machines. Use when the user says "sync memory", "save my memories", "push memory", or "pull memory from another machine". Memory lives in a separate PRIVATE git repo (RoboTwinMemory) nested at .claude/memory/. To sync memory AND conversations together, use /sync-claude.
---

# Sync memory

Project memory for this repo is stored in a **separate private git repo** nested at
`.claude/memory/` (remote `RoboTwinMemory`), gitignored by the public fork.
`.claude/hooks/sync-memory.sh` bridges that repo and this machine's live memory
store at `~/.claude/projects/<path-hash>/memory/`. See `.claude/README.md`.

Runs from a **login node** (network required); compute nodes are offline.

## Default action — SYNC (bidirectional: pull AND push)

```bash
bash .claude/hooks/sync-memory.sh sync
```

Exports this machine's live memory into `.claude/memory/`, commits, **pulls +
merges** memory from other machines, imports the merged set back into the live
store, and **always pushes**. Report per-store what happened.

## One-directional variants (only if asked)

```bash
bash .claude/hooks/sync-memory.sh save   # push-only: export + commit + push
bash .claude/hooks/sync-memory.sh pull   # pull-only: fetch + import to live store
```

## Guidance

- To sync memory together with conversations in one go, prefer `/sync-claude`.
- Merge caveat: syncing copies files additively (`cp -u`); a memory *deleted* on one
  machine isn't auto-deleted on another. To remove a memory everywhere, delete it in
  `.claude/memory/`, commit, and push.
- If the folder isn't a git repo (fresh clone), clone `RoboTwinMemory` into
  `.claude/memory/` first. Never move memory into the public main repo.
