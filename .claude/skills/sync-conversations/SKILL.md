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

To sync conversations AND memory together in one step, prefer `/sync-claude`.

Runs from a **login node** (network required); compute nodes are offline.

## Default action — SYNC (bidirectional: pull AND push)

Unless the user clearly wants one direction only, run the full bidirectional sync:

```bash
bash .claude/hooks/sync-conversations.sh sync
```

This exports this machine's live transcripts into `.claude/conversations/`, commits
them, **pulls + merges** any transcripts pushed from other machines, imports the
merged set back into the live store, and **always pushes** to `origin`. Report to
the user what was committed, whether the pull brought anything in, and that the push
succeeded. Every call ends with a push.

## One-directional variants (only if the user asks)

```bash
bash .claude/hooks/sync-conversations.sh save   # push-only: export + commit + push
bash .claude/hooks/sync-conversations.sh pull   # pull-only: fetch + import to live store
```

After a `pull`/`sync`, tell the user they can run `claude --resume` to pick up a
conversation recorded on another machine.

## Guidance

- If the script reports `.claude/conversations` is not a git repo, the user is on a
  fresh clone — instruct them to clone the private conversations repo into that
  folder first (`cd .claude/conversations` then `git clone <private-url> .`), per
  `.claude/README.md`.
- If push fails on auth, check `gh auth status` / SSH access to GitHub.
- Never move these transcripts into the main repo — it's a public fork.
