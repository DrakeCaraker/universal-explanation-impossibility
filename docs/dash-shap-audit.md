# dash-shap Repository Audit

Generated: 2026-04-02
Based on: Attribution Impossibility findings (188 theorems, 18 axioms, 0 sorry)

## Critical Bugs (fix before merge)

| # | Bug | Severity | File | Fix |
|---|-----|----------|------|-----|
| 1 | `_correlated_groups()` NOT transitive — misses chains A↔B↔C | **HIGH** | `stability.py:46-63`, `shap_pr/stability_check.py:153-165` | Replace linear scan with union-find |
| 2 | Multi-output SHAP not handled in `_compute_shap()` | **MEDIUM** | `stability.py:31-43` | Add `if isinstance(shap_values, list)` check |
| 3 | `validate()` checks all O(P²) pairs instead of correlated only | **LOW** | `stability.py:146-147` | Restrict to `_correlated_groups()` pairs |
| 4 | `consensus()` vs `core/consensus.py` compute different quantities | **LOW** | Both files | Document or align |
| 5 | Eager import of stability.py in `__init__.py` | **LOW** | `__init__.py:27` | Make lazy |
| 6 | CI coverage floor (69%) disagrees with pyproject.toml (70%) | **LOW** | `.github/workflows/ci.yml` | Align |

## Test Coverage: stability.py has ZERO coverage

- 342 tests total, 311 fast pass
- `screen()`, `validate()`, `consensus()`, `report()` — all untested
- Paper-demanded tests (flip≈50%, F1 |r|>0.8, DASH equity) — all missing

## Missing Features (ranked by impact/effort)

1. `recommend_M(sigma_sq, delta_sq)` → M_min formula (1 hr)
2. `rashomon_coefficient(X)` → dataset pre-screen (30 min)
3. `snr_diagnostic(results)` → per-pair SNR + predicted flip (2 hrs)
4. Impossibility warning in DASHPipeline.fit() (30 min)
5. `noise_estimate(model, X_test)` → KernelSHAP noise (2 hrs)
6. Group-level summary in report() (1 hr)
7. Progressive DASH adaptive M (1 day)

## PR #255 Blocking Issues

1. BUG 1 (transitive grouping) — ships wrong algorithm
2. BUG 2 (multi-output SHAP) — crashes classifiers
3. Zero test coverage for stability.py
4. Notebook implements everything inline, doesn't use library API

## Documentation Gaps

- README doesn't mention impossibility or link to dash-impossibility-lean
- No entry point for paper readers ("Coming from the impossibility paper?")
- impossibility_demo.ipynb doesn't use library API
- Two incompatible APIs (DASHPipeline vs stability.py) with no guidance on when to use which
- API_REFERENCE.md doesn't document stability.py

## Release Checklist

### Merge-blocking (~6 hrs)
- [ ] Fix _correlated_groups() union-find
- [ ] Fix _compute_shap() multi-output
- [ ] Add test_stability.py (≥10 tests)
- [ ] Rewrite impossibility_demo.ipynb to use API

### 0.2.0 release (~2 days)
- [ ] Lazy imports for stability.py
- [ ] Type hints on all functions
- [ ] recommend_M() + rashomon_coefficient()
- [ ] README impossibility section
- [ ] API docs for stability.py
- [ ] Fix CI coverage floor
- [ ] [dev] extras in pyproject.toml

### Full roadmap (~1 week)
- [ ] snr_diagnostic, noise_estimate
- [ ] Progressive DASH
- [ ] CLI diagnose command
- [ ] PyPI release
