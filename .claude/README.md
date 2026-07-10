# `.claude/` — project Claude Code config (checked into git)

This directory is version-controlled so every clone of this repo, on every
machine, gets the same Claude Code behavior regardless of the repo's absolute
path or the machine's GPU/deps.

## What's here

| Path | Purpose |
|---|---|
| `settings.json` | Project settings: default model `opus`, default mode `acceptEdits` ("auto"), custom status line, and conversation-sync hooks. |
| `statusline.sh` | Status line renderer → `auto · Opus 4.8 · ctx 45.1k/200k (23%)` (mode · model · context usage). |
| `hooks/sync-conversations.sh` | Syncs transcripts between the local store and `conversations/`. |
| `conversations/` | Conversation transcripts (`*.jsonl`), tracked in a **separate private git repo** — **gitignored** by this (public) repo. See below. |
| `skills/` | Project skills (auto-discovered). See `skills/README.md`. |

## Settings applied

- **Model:** `opus` (newest Opus) by default.
- **Mode:** `acceptEdits` — "auto" mode, so edits apply without a prompt. Cycle
  modes anytime with `Shift+Tab`.
- **Status line:** shows current mode, model, and live context-window usage.

## Resuming conversations across machines (without exposing them)

This repo's `origin` is a **public GitHub fork**, and transcripts leak absolute
paths, command output, and sometimes secrets — so they must **never** enter this
repo's history. Instead, `.claude/conversations/` is:

- **gitignored** by this repo (see `.gitignore`), and
- **its own separate private git repo** with its own private remote.

The main repo therefore has *zero* reference to conversations — no content, no
submodule, no URL. The two repos are fully decoupled.

Claude stores live transcripts in `~/.claude/projects/<path-hash>/`, where the hash
is derived from the repo's absolute path (differs per machine). The hooks bridge
that store and the nested `conversations/` repo:

- **`SessionStart` → `import`**: copies conversations into this machine's live
  store, so they appear in `claude --resume`.
- **`SessionEnd` → `export`**: copies this machine's transcripts into
  `conversations/` (as untracked files in the nested repo) ready to commit.

### First-time setup on a new machine

```bash
git clone <this repo>            # main repo — does NOT include conversations
cd <repo>/.claude
git clone <PRIVATE conversations remote> conversations   # the nested private repo
```

### Saving / syncing conversations (from a login node — needs network)

```bash
# export live -> nested repo, then commit + push the PRIVATE repo:
bash .claude/hooks/sync-conversations.sh export
cd .claude/conversations && git add -A && git commit -m "save conversations" && git push
```

On another machine: `cd .claude/conversations && git pull`, then start Claude once
(SessionStart imports them) and `claude --resume` lists them.

> Compute nodes are offline — only *record* there; commit/push the conversations
> repo from a login node. The `export`/`import` hooks are network-free file copies
> and run fine anywhere.
