# Co-Author Email Draft

**To**: Bryan Arnold, David Rhoads
**Subject**: The Attribution Impossibility — Ready for Review (JMLR submission)

---

Hi Bryan and David,

The Attribution Impossibility paper is ready for your review. This is Paper 3 in the dash-shap research program.

## What's Ready

- **Paper**: 65-page monograph (arXiv), 54-page JMLR submission, 10-page NeurIPS + 79-page supplement.
- **Lean formalization**: 305 theorems, 16 axioms, 0 sorry, 54 files. Builds with `lake build`.
- **Experiments**: 51 scripts, all reproducible on Apple Silicon in <30 min total.
- **Companion code**: single-model screen → multi-model Z-test → DASH workflow in [dash-shap PR #255](https://github.com/DrakeCaraker/dash-shap/pull/255).

## Documents to Read

1. **Start here**: [docs/co-author-guide.md](docs/co-author-guide.md) — plain English overview + 60-second summary
2. **Verification checklist**: [docs/verification-audit.md](docs/verification-audit.md) — 32-item ranked checklist
3. **Self-verification**: [docs/self-verification-report.md](docs/self-verification-report.md) — machine-verified results

## What I Need From You

1. **Read the full paper** (main.tex + at least the first 20 pages of supplement)
2. **Lean ↔ paper alignment**: For each numbered theorem in main.tex, verify the Lean statement matches the English
3. **Axiom plausibility**: Read the 16 axioms (6 type/constant, 6 property, 2 measure, 2 query) — are they reasonable idealizations?
4. **Read Laberge et al. (2023)**: Verify our impossibility is genuinely new (not already proved informally)
5. **Re-run 3 key experiments** on your machine: `validate_ratio.py`, `cross_implementation_validation.py`, `snr_calibration.py`

## Timeline

- **Feedback needed by**: April 21
- **JMLR submission target**: Q4 2026
- **arXiv preprint**: Before May 4

## Quick Start

```bash
git clone https://github.com/DrakeCaraker/dash-impossibility-lean.git
cd dash-impossibility-lean
lake build                    # ~5 min, verifies all 190 theorems
python3 paper/scripts/axiom_consistency_model.py  # verifies axiom system
```

Let me know if you have questions. The co-author guide has the full reading plan.

Drake
