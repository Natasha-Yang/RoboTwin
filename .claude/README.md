# `.claude/` — project Claude Code config (checked into git)

This directory is version-controlled so every clone of this repo, on every
machine, gets the same Claude Code behavior regardless of the repo's absolute
path or the machine's GPU/deps.

## What's here

| Path | Purpose |
|---|---|
| `settings.json` | Project settings: default model `opus`, default mode `acceptEdits` ("auto"), custom status line, and conversation/memory sync hooks. |
| `statusline.sh` / `.py` | Status line renderer → `auto · Opus 4.8 · ctx 45.1k/200k (23%) · /sync-claude` (mode · model · context usage · skill hint). |
| `hooks/sync-store.sh` | Generic sync engine (bidirectional git-backed sync of a Claude "store"). |
| `hooks/sync-conversations.sh`, `hooks/sync-memory.sh` | Thin wrappers over `sync-store.sh` for the two stores. |
| `conversations/` | Conversation transcripts (`*.jsonl`) — **separate private git repo**, **gitignored** here. |
| `memory/` | Project memory (`*.md`, incl. `MEMORY.md`) — **separate private git repo**, **gitignored** here. |
| `skills/` | Project skills (auto-discovered): `sync-claude`, `sync-conversations`, `sync-memory`. See `skills/README.md`. |

## Settings applied

- **Model:** `opus` (newest Opus) by default.
- **Mode:** `acceptEdits` — "auto" mode, so edits apply without a prompt. Cycle
  modes anytime with `Shift+Tab`.
- **Status line:** shows current mode, model, live context-window usage, and the
  `/sync-claude` reminder.

## Syncing portable Claude state across machines (without exposing it)

This repo's `origin` is a **public GitHub fork**, and both conversation transcripts
and memory can leak absolute paths, command output, and sometimes secrets — so they
must **never** enter this repo's history. Each is kept as its **own separate private
git repo**, nested here and **gitignored** by this repo:

| Store | Nested path | Live store on each machine | Private remote |
|---|---|---|---|
| conversations | `.claude/conversations/` | `~/.claude/projects/<path-hash>/*.jsonl` | `RoboTwinConvos` |
| memory | `.claude/memory/` | `~/.claude/projects/<path-hash>/memory/*.md` | `RoboTwinMemory` |

The public repo has *zero* reference to either — no content, no submodule, no URL.
The `<path-hash>` is derived from the repo's absolute path, so it differs per
machine; the hooks bridge that machine-specific live store and the portable nested
repos:

- **`SessionStart` → `import`** (both stores): copies into this machine's live store
  (conversations appear in `claude --resume`; memory is live for new sessions).
- **`SessionEnd` → `export`** (both stores): copies this machine's live state into
  the nested repos, ready to commit. (Network-free file copies.)

### Everyday sync (from a login node — needs network)

Run the **`/sync-claude`** skill, or:

```bash
bash .claude/hooks/sync-conversations.sh sync   # bidirectional: pull + push
bash .claude/hooks/sync-memory.sh sync          # bidirectional: pull + push
```

`sync` = export live → commit → pull/merge other machines' state → import → **push**.
Per-store `save` (push-only) and `pull` (pull-only) variants also exist.

### First-time setup on a new machine

The main clone contains **neither** store; clone each private repo into place:

```bash
git clone git@github.com:Natasha-Yang/RoboTwin.git && cd RoboTwin
cd .claude
git clone git@github.com:Natasha-Yang/RoboTwinConvos.git  conversations
git clone git@github.com:Natasha-Yang/RoboTwinMemory.git  memory
cd .. && bash .claude/hooks/sync-conversations.sh pull && bash .claude/hooks/sync-memory.sh pull
```

> Compute nodes are offline — only *record* there; commit/push from a login node.
> The `export`/`import` hooks are network-free and run fine anywhere.
> `gh`'s API can't see these private repos (fine-grained PAT scope), but the sync
> uses git-over-SSH, which works.
