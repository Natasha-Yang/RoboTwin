---
name: sync-conversations
description: Save, back up, or sync Claude Code conversation transcripts for this repo across machines. Use when the user says "sync conversations", "save my conversations", "back up this chat", "push conversations", or "pull conversations from another machine". Conversations live in a separate PRIVATE git repo nested at .claude/conversations/.
---

# Sync conversations

Conversation transcripts for this repo are stored in a **separate private git
repo** nested at `.claude/conversations/` (gitignored by the main public fork).
`.claude/hooks/sync-conversations.sh` bridges that repo and the machine's live
store at `~/.claude/projects/<path-hash>/`. See `.claude/README.md` for the full
design.

Both actions must run from a **login node** (network required for push/pull); the
compute nodes are offline.

## Default action — SAVE (export + commit + push)

Run this to back up the current machine's conversations to the private remote:

```bash
bash .claude/hooks/sync-conversations.sh save
```

This exports the live transcripts into `.claude/conversations/`, commits them
(timestamp + hostname message, no-op if nothing changed), and pushes to `origin`.
Report to the user whether anything was committed and whether the push succeeded.

## PULL (fetch + import) — bring in conversations from other machines

When the user wants conversations recorded elsewhere to be resumable here:

```bash
bash .claude/hooks/sync-conversations.sh pull
```

This pulls the latest from the private remote and imports every transcript into
this machine's live store. Afterward tell the user they can run `claude --resume`
to pick a conversation.

## Guidance

- If the script reports `.claude/conversations` is not a git repo, the user is on a
  fresh clone — instruct them to clone the private conversations repo into that
  folder first (`cd .claude/conversations` then `git clone <private-url> .`), per
  `.claude/README.md`.
- If push fails on auth, check `gh auth status` / SSH access to GitHub.
- Never move these transcripts into the main repo — it's a public fork.
