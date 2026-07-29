# Archive

Not maintained; not covered by `make ci`. Kept for provenance only. The live,
CI-protected papers are in `paper/` (see below).

## Monograph provenance
- `universal_impossibility_monograph_v1_original.tex` — the pre-restructure monograph.
- `restructure_monograph_v2.py` — the deterministic generator that produced the tiered
  v2 from it (source paths need adjusting to reproduce). The canonical, directly-edited
  monograph now lives at `paper/universal_impossibility_monograph.tex`.

## Non-live venue drafts (archived 2026-07-29)
Superseded or non-selected venue variants. They carried older-era counts/architecture
descriptions; they were left as-is on archiving rather than re-verified. Do not submit
without a fresh consistency pass.
- `universal_impossibility.tex` — superseded base version.
- `universal_impossibility_pnas.tex`, `universal_impossibility_pnas_si.tex` — PNAS variant
  (not selected; Nature is the flagship).
- `universal_impossibility_neurips.tex`, `universal_impossibility_neurips_supplement.tex` —
  NeurIPS variant (not selected).
- `nature_brief_communication.tex` — Nature brief-communication variant (not selected).
- `nature_si_cover.tex` — SI cover for the brief communication (follows it).
- `explanation_capacity_audit.tex` — standalone audit; its ρ* sensitivity sweep is folded
  into the monograph's capacity-audit section.

## Live papers (in `paper/`, CI-protected)
- `universal_impossibility_monograph.tex` — canonical arXiv archival document (source of truth).
- `nature_article.tex` + `supplementary_information.tex` + `nature_cover_letter.tex` — Nature flagship.
- `universal_impossibility_jmlr.tex` — JMLR full technical version.
- `nature_comment_regulatory.tex` — policy comment (EU AI Act framing).
- `claims.tex` (generated counts) and `lean_appendix.tex` are shared infrastructure.
