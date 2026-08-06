# The Limits of Explanation — interactive explainer

Source of record for the live teaching page and its content draft.

| File | Role |
| --- | --- |
| `limits-of-explanation.html` | The deployed page, self-contained (inline CSS/JS, no external requests). |
| `EXPLAINER_ONBOARDING_FINAL.md` | Content source of truth for the *narrative*. **Read the changelogs at the end first** — see the divergence warning below. |

## Live artifact

- URL: <https://claude.ai/code/artifact/358ef91a-2e0c-4cc0-82ce-22312421211e>
- `<title>`: "The Limits of Explanation" · favicon: 🔬 (keep both stable across redeploys)
- Redeploying from a new Claude Code session **requires passing this URL as the `url` parameter** to the Artifact tool — otherwise a fresh conversation mints a new artifact URL.

## Design decisions (locked — do not relitigate without Drake)

- The page is a **teaching-only cut**. The falsification/credibility program (former node 10: η-law confession, self-refutations, Falsification Ledger, W4, preregistration material, Retired tier, frozen-bet opener, hero receipts chips) lives in the papers, not on the page. The only pointer is the neutral footer line. Never reintroduce falsification language.
- 12 questions (nodes 0–11); ribbon phases: The Ceiling 1–6 · The Instrument 7–9 · The Frontier 10 · The Ask 11. Three widgets (W1 pick-two @2, W2 how-often @6, W3 slider @9).
- Minimal hero: H1 + byline only — no epigraph, deck, or chips.

## Draft-vs-page divergence

`EXPLAINER_ONBOARDING_FINAL.md` deliberately describes **more than the page shows**: its body keeps the full 13-node content (node 10, falsifications) as the paper-side narrative. The changelog entries at its end are the authoritative record of every page divergence, including the teaching cut. Do **not** "fix" the page from the draft body.

## Editing gotchas

- Widget state lives in `sessionStorage` under `loe_state`. Ribbon stamps for Q6/Q9 are gated until the widget commits (spoiler fix) — preserve that when touching JS.
- Repo deep links on the page are pinned to commit `c74d82d`. Monograph "§NNNN" line anchors stay valid only while `paper/universal_impossibility_monograph.tex` edits are line-count-neutral (5,854 lines as of `c9cd389`); a future pin bump must re-verify every anchor.
- Lean counts quoted on the page (130 files / 715 theorems / 4 axioms, etc.) drift with any Lean merge — re-verify against `paper/claims.tex` before touching them. Two standing overclaim traps: the phase-problem Lean is a scalar toy (never "machine-checked crystallography"); Arrow's *setting* is formalized, not the 1951 theorem.
