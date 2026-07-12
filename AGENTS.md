# PairwiseTurnGPT

<!-- vault:shared:begin -- generated from `99 Templates/Project AGENTS.md`; edits here are overwritten by `vault-project sync` -->
## Project memory

This repo's working memory lives at the repo root, **symlinked from the second-brain vault**
(canonical copy in `~/Projects/Nordorn/07 Projects/<name>/`). Edit through the symlink —
never fork a copy into the repo. A forked copy is invisible to the vault, and the wiki
cannot reason about what it cannot see. The repo gitignores these paths for that reason.

- `Notes.md`     — current truth: overview, architecture, subsystem map.
- `LOG.md`       — append-only session log, newest first.
- `Todo.md`      — the build order. Keep it honest each session.
- `PLAN.md`      — active plan for in-flight work (absent when nothing's in flight).
- `DECISIONS.md` — load-bearing decisions; each carries the constraint it implies.

## The vault is reachable from this repo

`.vault/` is a symlink to the second-brain vault root (gitignored — local only). Every vault
path in this file resolves through it, so read them directly rather than guessing:

- `.vault/04 Indexes/Working with Claude.md` — how the vault operates. **Read this before
  writing anything into the vault.**
- `.vault/CLAUDE.md` — the vault's binding contract (raw layers are read-only; Base Notes are
  proposed, never silently created).
- `.vault/00 Vault Guide.md` — folder purposes, capture pipeline, note types.
- `.vault/08 Trackers/Current WIP.md` — the board. What the user is actually meant to do next.
- `.vault/04 Indexes/Wiki Index.md` — catalog of what the wiki already knows. Check here
  before concluding something is a new idea.
- `.vault/05 Base Notes/` — synthesised knowledge. Link to a Base Note rather than restating it.

The vault's rules bind you when you write into it: don't touch `01 Fleeting Notes/`,
`02 Sources/` or `06 Daily Notes/` (raw, read-only), and *propose* Base Notes rather than
creating them silently.

## Session start — read before acting

1. Read `Notes.md`, the top 2–3 `LOG.md` entries, and `PLAN.md` if present.
2. Skim `DECISIONS.md` — the **Constraint** lines are the do/don't rules not to violate.
3. The latest `## Open / next` in `LOG.md` is the starting point.
4. Check for divergence before building on top: unpushed commits, un-pulled remote changes,
   uncommitted files from a prior session.

Don't re-derive what's already written. If `PLAN.md` and the ask disagree, confirm first.

## Session end / before auto-compact — write before stopping

1. Prepend or refresh a dated `LOG.md` entry (What changed / Decisions / Learnings / Open-next),
   written for a reader who wasn't there. Do this *before* compaction, not only at session end.
2. Update `Notes.md` if state or architecture changed (propose the diff; the user owns it),
   and check off / add to `Todo.md`.
3. A load-bearing decision → `### D-NNN` in `DECISIONS.md`, with its **Referent** and
   **Constraint** lines. Supersede, never delete.
4. A learning useful on a *different* project → say so and offer to promote it to the vault.
   Project work that traps generalisable knowledge in the project folder is a leak.
5. If this session changed the project's *status* — a kill, a falsification, a shipped premise —
   update `08 Trackers/Current WIP.md` in the vault in the same pass. The log is history;
   the board is the instruction. A finding that doesn't reach the board doesn't exist.
6. Commit (one line, no `Co-Authored-By` trailer) and push if a remote exists.

These memory files are not code — touching them never requires touching the codebase.
Full convention: `04 Indexes/Working with Claude.md` in the vault.

## This region is generated — don't edit it here

Everything between the `vault:shared` markers is rendered from `99 Templates/Project AGENTS.md`
in the vault and is overwritten by `scripts/vault-project sync`. To change what *every* project
tells its agent, edit the template. Project-specific instructions go **below the closing
marker**, where they survive syncs.

`AGENTS.md` is the canonical file. `CLAUDE.md` is a symlink to it so Claude Code, Codex,
Cursor, Gemini CLI and friends all read the same instructions.
<!-- vault:shared:end -->

## Purpose

*What this project is. Written by hand.*
