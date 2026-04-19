#!/usr/bin/env python3
"""
Cross-Domain Noether Counting Test

Tests whether the symmetry group structure predicts WHICH conclusions
are stable vs unstable in three multi-analyst studies:

1. NARPS: functional networks → regional disagreement patterns (d=0.32, prior result)
2. Silberzahn: analytic method → estimate agreement (ANOVA η²)
3. Breznau: team identity → model agreement (ICC)

The Noether prediction: within-group comparisons are unstable,
between-group conclusions are stable. The fraction of variance
explained by the grouping (η² or ICC) measures how much the
symmetry structure organizes the disagreement.
"""

import warnings
warnings.filterwarnings('ignore')

import json, time, os
import numpy as np
from scipy.stats import f_oneway, mannwhitneyu, kruskal
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_BOOTSTRAP = 2000


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def categorize_silberzahn_method(approach):
    """Categorize analytic approach into broad method families.

    Categories chosen to be defensible and non-arbitrary:
    - Logistic/binomial: binary outcome models
    - Linear: OLS, linear probability, WLS
    - Poisson/count: Poisson, negative binomial, zero-inflated
    - Bayesian: explicitly Bayesian methods
    - Other: correlation, Tobit, etc.
    """
    a = approach.lower()
    if any(w in a for w in ['logistic', 'logit', 'binomial', 'log-linear']):
        return 'Logistic'
    elif any(w in a for w in ['poisson', 'negative binomial', 'zero-inflated', 'count']):
        return 'Count'
    elif any(w in a for w in ['linear probability', 'ordinary least squares', 'linear regression',
                               'multiple linear', 'weighted least squares']):
        return 'Linear'
    elif any(w in a for w in ['bayesian', 'bayes', 'dirichlet']):
        return 'Bayesian'
    else:
        return 'Other'


def test_silberzahn_noether():
    """Noether test for Silberzahn: method type → estimate agreement."""
    print("\n" + "=" * 60)
    print("SILBERZAHN: METHOD TYPE → ESTIMATE AGREEMENT")
    print("=" * 60)

    # Load data
    path = os.path.join(SCRIPT_DIR, 'silberzahn_2018_team_estimates.csv')
    teams = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            teams.append({
                'team': row['Team'],
                'approach': row['Analytic.Approach'],
                'log_or': np.log(float(row['OR'])),
                'category': categorize_silberzahn_method(row['Analytic.Approach']),
            })

    # Show categorization
    categories = {}
    for t in teams:
        categories.setdefault(t['category'], []).append(t)

    print(f"\n  Method categories (n={len(teams)} teams):")
    for cat in sorted(categories):
        members = categories[cat]
        estimates = [t['log_or'] for t in members]
        print(f"    {cat:>10s}: n={len(members):2d}, mean log(OR)={np.mean(estimates):.3f}, "
              f"std={np.std(estimates):.3f}")

    # One-way ANOVA: category → log(OR)
    groups = [np.array([t['log_or'] for t in members])
              for members in categories.values()
              if len(members) >= 2]
    group_labels = [cat for cat, members in categories.items() if len(members) >= 2]

    if len(groups) >= 2:
        F, anova_p = f_oneway(*groups)
        H, kw_p = kruskal(*groups)
    else:
        F, anova_p = 0, 1
        H, kw_p = 0, 1

    # η² (effect size)
    all_estimates = np.array([t['log_or'] for t in teams])
    grand_mean = np.mean(all_estimates)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_estimates - grand_mean)**2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    print(f"\n  ANOVA: F={F:.2f}, p={anova_p:.4f}")
    print(f"  Kruskal-Wallis: H={H:.2f}, p={kw_p:.4f}")
    print(f"  η² = {eta_sq:.3f} (variance explained by method type)")

    # Bootstrap CI on η²
    rng = np.random.RandomState(42)
    boot_eta = []
    n = len(teams)
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, n, replace=True)
        boot_teams = [teams[i] for i in idx]
        boot_cats = {}
        for t in boot_teams:
            boot_cats.setdefault(t['category'], []).append(t['log_or'])
        boot_groups = [np.array(v) for v in boot_cats.values() if len(v) >= 2]
        if len(boot_groups) >= 2:
            boot_all = np.concatenate(boot_groups)
            boot_grand = np.mean(boot_all)
            boot_ss_b = sum(len(g) * (np.mean(g) - boot_grand)**2 for g in boot_groups)
            boot_ss_t = np.sum((boot_all - boot_grand)**2)
            boot_eta.append(boot_ss_b / boot_ss_t if boot_ss_t > 0 else 0)
    boot_eta = np.array(boot_eta)
    eta_ci = [float(np.percentile(boot_eta, 2.5)), float(np.percentile(boot_eta, 97.5))]
    print(f"  η² CI: [{eta_ci[0]:.3f}, {eta_ci[1]:.3f}]")

    # Within-method vs between-method |Δestimate| (parallel to NARPS Noether)
    within_diffs = []
    between_diffs = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            diff = abs(teams[i]['log_or'] - teams[j]['log_or'])
            if teams[i]['category'] == teams[j]['category']:
                within_diffs.append(diff)
            else:
                between_diffs.append(diff)

    within_diffs = np.array(within_diffs)
    between_diffs = np.array(between_diffs)

    mean_within = float(np.mean(within_diffs))
    mean_between = float(np.mean(between_diffs))
    pooled = np.sqrt((np.var(within_diffs) * len(within_diffs) +
                      np.var(between_diffs) * len(between_diffs)) /
                     (len(within_diffs) + len(between_diffs)))
    cohens_d = (mean_between - mean_within) / pooled if pooled > 0 else 0

    U, mw_p = mannwhitneyu(within_diffs, between_diffs, alternative='less')

    print(f"\n  WITHIN vs BETWEEN method |Δestimate|:")
    print(f"    Within-method: {mean_within:.4f} (n={len(within_diffs)})")
    print(f"    Between-method: {mean_between:.4f} (n={len(between_diffs)})")
    print(f"    Cohen's d: {cohens_d:.3f}")
    print(f"    Mann-Whitney p (within < between): {mw_p:.4f}")

    # Permutation null: shuffle method labels
    observed_gap = mean_between - mean_within
    rng2 = np.random.RandomState(99)
    perm_gaps = []
    cat_labels = [t['category'] for t in teams]
    for _ in range(5000):
        perm_labels = rng2.permutation(cat_labels)
        w, b = [], []
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                diff = abs(teams[i]['log_or'] - teams[j]['log_or'])
                if perm_labels[i] == perm_labels[j]:
                    w.append(diff)
                else:
                    b.append(diff)
        perm_gaps.append(np.mean(b) - np.mean(w))
    perm_gaps = np.array(perm_gaps)
    perm_p = float(np.mean(perm_gaps >= observed_gap))

    print(f"    Permutation p: {perm_p:.4f}")
    print(f"    Null gap: {np.mean(perm_gaps):.4f} ± {np.std(perm_gaps):.4f}")

    confirmed = mw_p < 0.05 and perm_p < 0.05
    print(f"\n  NOETHER PREDICTION: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")
    if confirmed:
        print(f"    Teams using similar methods agree more than teams using different methods")
    else:
        print(f"    Method type does NOT significantly predict agreement")

    return {
        'n_teams': len(teams),
        'n_categories': len(categories),
        'categories': {cat: len(members) for cat, members in categories.items()},
        'eta_squared': float(eta_sq),
        'eta_squared_ci': eta_ci,
        'anova_F': float(F), 'anova_p': float(anova_p),
        'kruskal_H': float(H), 'kruskal_p': float(kw_p),
        'within_mean': mean_within, 'between_mean': mean_between,
        'cohens_d': float(cohens_d),
        'mann_whitney_p': float(mw_p),
        'permutation_p': perm_p,
        'confirmed': confirmed,
    }


def test_breznau_icc():
    """Noether test for Breznau: team identity → model agreement (ICC)."""
    print("\n" + "=" * 60)
    print("BREZNAU: TEAM IDENTITY → MODEL AGREEMENT (ICC)")
    print("=" * 60)

    # Load all-models data
    path = os.path.join(SCRIPT_DIR, 'breznau_2022_all_models.csv')
    models = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                models.append({
                    'team': int(row['team_id']),
                    'model': row['model_id'],
                    'ame': float(row['AME']),
                })
            except (ValueError, KeyError):
                pass

    # Group by team
    teams = {}
    for m in models:
        teams.setdefault(m['team'], []).append(m['ame'])

    # Filter to teams with >= 2 models
    teams = {t: np.array(v) for t, v in teams.items() if len(v) >= 2}
    n_teams = len(teams)
    n_models = sum(len(v) for v in teams.values())

    print(f"\n  {n_teams} teams, {n_models} total models")
    print(f"  Models per team: {min(len(v) for v in teams.values())} to {max(len(v) for v in teams.values())}")

    # ICC(1,1) — one-way random effects
    # ICC = (MSB - MSW) / (MSB + (k-1)*MSW)
    # where k = mean group size, MSB = between-group mean square, MSW = within-group mean square
    all_ame = np.concatenate(list(teams.values()))
    grand_mean = np.mean(all_ame)

    # Between-group sum of squares
    ss_between = sum(len(v) * (np.mean(v) - grand_mean)**2 for v in teams.values())
    df_between = n_teams - 1

    # Within-group sum of squares
    ss_within = sum(np.sum((v - np.mean(v))**2) for v in teams.values())
    df_within = n_models - n_teams

    msb = ss_between / df_between if df_between > 0 else 0
    msw = ss_within / df_within if df_within > 0 else 1

    # Harmonic mean of group sizes
    k_sizes = [len(v) for v in teams.values()]
    k_harmonic = len(k_sizes) / sum(1 / k for k in k_sizes)

    icc = (msb - msw) / (msb + (k_harmonic - 1) * msw) if (msb + (k_harmonic - 1) * msw) > 0 else 0

    # F-test for ICC significance
    F_icc = msb / msw if msw > 0 else 0
    from scipy.stats import f as f_dist
    icc_p = 1 - f_dist.cdf(F_icc, df_between, df_within)

    # η² from the one-way structure
    eta_sq = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else 0

    print(f"\n  ICC(1,1) = {icc:.3f}")
    print(f"  F = {F_icc:.1f}, p = {icc_p:.2e}")
    print(f"  η² = {eta_sq:.3f}")
    print(f"  MSB (between-team) = {msb:.6f}")
    print(f"  MSW (within-team) = {msw:.6f}")
    print(f"  MSB/MSW ratio = {msb/msw:.1f}" if msw > 0 else "")

    # Bootstrap CI on ICC
    rng = np.random.RandomState(42)
    boot_icc = []
    team_ids = list(teams.keys())
    for _ in range(N_BOOTSTRAP):
        # Resample teams (cluster bootstrap)
        boot_team_ids = rng.choice(team_ids, len(team_ids), replace=True)
        boot_all = []
        boot_teams_data = {}
        for i, tid in enumerate(boot_team_ids):
            boot_teams_data[i] = teams[tid]
            boot_all.extend(teams[tid])
        boot_all = np.array(boot_all)
        boot_grand = np.mean(boot_all)
        n_b_models = sum(len(v) for v in boot_teams_data.values())

        bss_b = sum(len(v) * (np.mean(v) - boot_grand)**2 for v in boot_teams_data.values())
        bss_w = sum(np.sum((v - np.mean(v))**2) for v in boot_teams_data.values())
        bdf_b = len(boot_teams_data) - 1
        bdf_w = n_b_models - len(boot_teams_data)

        if bdf_b > 0 and bdf_w > 0:
            bmsb = bss_b / bdf_b
            bmsw = bss_w / bdf_w
            bk = len(boot_teams_data) / sum(1 / len(v) for v in boot_teams_data.values())
            bicc = (bmsb - bmsw) / (bmsb + (bk - 1) * bmsw) if (bmsb + (bk - 1) * bmsw) > 0 else 0
            boot_icc.append(bicc)

    boot_icc = np.array(boot_icc)
    icc_ci = [float(np.percentile(boot_icc, 2.5)), float(np.percentile(boot_icc, 97.5))]

    print(f"  ICC CI: [{icc_ci[0]:.3f}, {icc_ci[1]:.3f}]")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    print(f"    ICC = {icc:.3f} means {icc*100:.0f}% of model-level AME variance is")
    print(f"    between teams (methodological approach), {(1-icc)*100:.0f}% is within")
    print(f"    teams (model specification). The team's approach IS the dominant")
    print(f"    source of variation {'✓' if icc > 0.3 else '✗'}")

    # Within-team vs between-team |ΔAME| (parallel to NARPS/Silberzahn)
    team_means = {t: np.mean(v) for t, v in teams.items()}
    within_diffs = []
    between_diffs = []

    team_id_list = list(teams.keys())
    # Within-team: pairs of models from the same team
    for tid in team_id_list:
        vals = teams[tid]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                within_diffs.append(abs(vals[i] - vals[j]))

    # Between-team: pairs of team means
    for i in range(len(team_id_list)):
        for j in range(i + 1, len(team_id_list)):
            between_diffs.append(abs(team_means[team_id_list[i]] - team_means[team_id_list[j]]))

    within_diffs = np.array(within_diffs)
    between_diffs = np.array(between_diffs)

    print(f"\n  Within-team mean |ΔAME|: {np.mean(within_diffs):.4f} (n={len(within_diffs)})")
    print(f"  Between-team mean |Δmean_AME|: {np.mean(between_diffs):.4f} (n={len(between_diffs)})")

    confirmed = icc > 0.1 and icc_p < 0.05
    print(f"\n  NOETHER PREDICTION: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")
    if confirmed:
        print(f"    Team methodology explains {icc*100:.0f}% of model variation")
        print(f"    Within-team model specification is the unstable part")

    return {
        'n_teams': n_teams,
        'n_models': n_models,
        'icc': float(icc),
        'icc_ci': icc_ci,
        'icc_F': float(F_icc),
        'icc_p': float(icc_p),
        'eta_squared': float(eta_sq),
        'msb_msw_ratio': float(msb / msw) if msw > 0 else 0,
        'confirmed': confirmed,
    }


def main():
    start = time.time()
    print("=" * 60)
    print("CROSS-DOMAIN NOETHER COUNTING TEST")
    print("Does the symmetry group predict which conclusions are stable?")
    print("=" * 60)

    results = {}

    results['silberzahn'] = test_silberzahn_noether()
    results['breznau'] = test_breznau_icc()

    elapsed = time.time() - start

    # Cross-domain synthesis
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN NOETHER SYNTHESIS")
    print(f"{'='*60}")

    # NARPS reference (from prior definitive analysis)
    narps_eta_sq = 0.354  # from test_network_identity in eta_approx script
    narps_d = 0.32  # from definitive activation-controlled test
    narps_confirmed = True

    print(f"\n  {'Study':>30s} {'Group structure':>20s} {'η² or ICC':>10s} {'d':>8s} {'Confirmed':>10s}")
    print("  " + "-" * 85)
    print(f"  {'NARPS (neuroscience)':>30s} {'Functional networks':>20s} {narps_eta_sq:10.3f} {narps_d:8.3f} {'YES':>10s}")

    s = results['silberzahn']
    print(f"  {'Silberzahn (psychology)':>30s} {'Analytic method':>20s} {s['eta_squared']:10.3f} {s['cohens_d']:8.3f} "
          f"{'YES' if s['confirmed'] else 'NO':>10s}")

    b = results['breznau']
    print(f"  {'Breznau (political science)':>30s} {'Team methodology':>20s} {b['icc']:10.3f} {'N/A':>8s} "
          f"{'YES' if b['confirmed'] else 'NO':>10s}")

    n_confirmed = sum([narps_confirmed, s['confirmed'], b['confirmed']])
    print(f"\n  CONFIRMED: {n_confirmed}/3 domains")

    if n_confirmed == 3:
        print(f"\n  ✓ KNOCKOUT: The Noether counting prediction holds across ALL three domains.")
        print(f"    The symmetry group structure (networks / methods / teams) predicts")
        print(f"    which conclusions are stable in neuroscience, psychology, and political science.")
    elif n_confirmed >= 2:
        print(f"\n  ⚠ PARTIAL: Noether prediction holds in {n_confirmed}/3 domains.")
    else:
        print(f"\n  ✗ The Noether prediction does not generalize across domains.")

    print(f"\n  WHAT THIS MEANS:")
    print(f"    Not just 'disagreement exists' (known) or 'averaging helps' (known)")
    print(f"    but 'HERE IS which conclusions survive and which don't' (NEW)")
    print(f"    — predicted by the same mathematical structure across three fields.")

    print(f"\n  Elapsed: {elapsed:.0f}s")

    output = {
        'experiment': 'noether_cross_domain',
        'results': results,
        'narps_reference': {'eta_squared': narps_eta_sq, 'd': narps_d, 'confirmed': True},
        'n_confirmed': n_confirmed,
        'elapsed_seconds': elapsed,
    }

    out_path = os.path.join(SCRIPT_DIR, 'results_noether_cross_domain.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
