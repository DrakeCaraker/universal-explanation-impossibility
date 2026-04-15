#!/usr/bin/env python3
"""
Regulatory Compliance Metrics: What fraction of explanation is achievable?

For each of the 15 bridge datasets, compute:
1. Total pairwise comparisons
2. Structurally reliable (SNR > 2): can be reported with confidence
3. Marginal (0.5 < SNR < 2): report with uncertainty quantification
4. Structurally unreliable (SNR < 0.5): must be disclosed as indeterminate
5. Compliance score: fraction of explanation that IS achievable = reliable/total

This quantifies the "explainability gap" — the difference between what
regulations demand (100% explanation) and what is structurally achievable.
"""

import json
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def main():
    print("Regulatory Compliance Metrics")
    print("=" * 60)

    # Load the expanded bridge results
    bridge_path = OUT_DIR / 'results_explanation_landscape_bridge_expanded.json'
    if not bridge_path.exists():
        print("ERROR: Run explanation_landscape_bridge_expanded.py first")
        return

    bridge = json.load(open(bridge_path))
    datasets = bridge['per_dataset']

    # Also load clinical SHAP audit if available
    clinical_path = OUT_DIR / 'results_clinical_shap_audit.json'
    clinical = json.load(open(clinical_path)) if clinical_path.exists() else None

    print("\n  Compliance Summary (from bridge experiment, gain-based importance):\n")
    print(f"  {'Dataset':<25s} {'P':>3s} {'Reliable':>10s} {'Marginal':>10s} {'Unreliable':>10s} {'Compliance':>10s}")
    print(f"  {'-'*68}")

    all_reliable = 0
    all_marginal = 0
    all_unreliable = 0
    all_total = 0

    compliance_data = {}

    for name, r in datasets.items():
        cc_by_threshold = r.get('cc_by_threshold', {})

        # Use SNR thresholds from coverage conflict
        # cc at t=2.0 = fraction with SNR < 2.0 (unreliable + marginal)
        # cc at t=0.5 = fraction with SNR < 0.5 (unreliable)
        pct_unreliable = r.get('coverage_conflict_degree', 0) * 100  # SNR < 0.5
        pct_reliable = r.get('reliable_fraction', 0) * 100           # SNR > 2.0
        pct_marginal = 100 - pct_reliable - pct_unreliable

        n_pairs = r.get('n_pairs', 0)
        n_reliable = int(round(pct_reliable / 100 * n_pairs))
        n_marginal = int(round(pct_marginal / 100 * n_pairs))
        n_unreliable = int(round(pct_unreliable / 100 * n_pairs))

        all_reliable += n_reliable
        all_marginal += n_marginal
        all_unreliable += n_unreliable
        all_total += n_pairs

        compliance = pct_reliable

        compliance_data[name] = {
            "n_features": r.get("n_features", 0),
            "n_pairs": n_pairs,
            "pct_reliable": round(pct_reliable, 1),
            "pct_marginal": round(pct_marginal, 1),
            "pct_unreliable": round(pct_unreliable, 1),
            "compliance_score": round(compliance, 1),
            "mean_instability": r.get("mean_instability", 0),
        }

        print(f"  {name:<25s} {int(r.get('n_features', 0)):>3d} "
              f"{pct_reliable:>9.1f}% {pct_marginal:>9.1f}% "
              f"{pct_unreliable:>9.1f}% {compliance:>9.1f}%")

    # Aggregate
    agg_reliable = all_reliable / all_total * 100 if all_total > 0 else 0
    agg_marginal = all_marginal / all_total * 100 if all_total > 0 else 0
    agg_unreliable = all_unreliable / all_total * 100 if all_total > 0 else 0

    print(f"  {'-'*68}")
    print(f"  {'AGGREGATE':<25s} {'':>3s} "
          f"{agg_reliable:>9.1f}% {agg_marginal:>9.1f}% "
          f"{agg_unreliable:>9.1f}% {agg_reliable:>9.1f}%")

    # Clinical SHAP audit (TreeSHAP-based)
    if clinical:
        print(f"\n\n  Clinical Audit (TreeSHAP, top-5 features only):\n")
        print(f"  {'Dataset':<25s} {'Top-5 Reliable':>15s} {'Top-5 Unreliable':>15s}")
        print(f"  {'-'*55}")
        for name, r in clinical['per_dataset'].items():
            t5 = r['top_5_audit']
            print(f"  {name:<25s} {t5['n_reliable']:>7d}/10     "
                  f"{t5['n_unreliable']:>7d}/10")

    # Headline
    print(f"\n\n  HEADLINE METRICS:")
    print(f"  Across 15 datasets (gain-based importance):")
    print(f"    Only {agg_reliable:.0f}% of feature comparisons are structurally reliable")
    print(f"    {agg_unreliable:.0f}% are structurally unreliable (coin flips)")
    print(f"    {agg_marginal:.0f}% are in the gray zone")
    print(f"  The 'explainability gap' = {100 - agg_reliable:.0f}% of explanation")
    print(f"  content cannot be guaranteed under current regulatory standards.")

    if clinical:
        print(f"\n  Across 5 clinical datasets (TreeSHAP, top-5 features):")
        print(f"    {clinical['headline_pct_unreliable_top5']}% of top-5 ranking "
              f"comparisons are unreliable")

    # Save
    output = {
        "experiment": "regulatory_compliance_metrics",
        "description": "Fraction of explanation achievable under impossibility",
        "aggregate": {
            "pct_reliable": round(agg_reliable, 1),
            "pct_marginal": round(agg_marginal, 1),
            "pct_unreliable": round(agg_unreliable, 1),
            "explainability_gap_pct": round(100 - agg_reliable, 1),
        },
        "per_dataset": compliance_data,
    }

    json_path = OUT_DIR / 'results_regulatory_compliance.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {json_path}")


if __name__ == '__main__':
    main()
