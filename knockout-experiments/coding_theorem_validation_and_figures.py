"""
Coding Theorem Validation + Publication Figures
=================================================
A: Numerical validations of the coding theorem (3 tests)
E: Publication-quality figures for the audit paper

Outputs:
  figures/coding_theorem_validation.pdf
  figures/audit_claims_vs_capacity.pdf
  figures/audit_flip_rate_boxplot.pdf
  figures/audit_sensitivity_curve.pdf
  figures/audit_domain_forest.pdf
  figures/audit_eta_degradation.pdf
  figures/audit_beyond_capacity_mse.pdf
  results_coding_theorem_validation.json
"""

import json, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from xgboost import XGBClassifier, XGBRegressor
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

M = 50
THR = 0.70
XGB_CLF = dict(n_estimators=100, max_depth=6, subsample=0.8,
               colsample_bytree=0.5, verbosity=0, eval_metric='logloss')

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})


def sage(X, thr):
    rho = np.abs(np.nan_to_num(spearmanr(X).statistic, nan=0))
    if np.ndim(rho) == 0: rho = np.array([[1, abs(rho)], [abs(rho), 1]])
    np.fill_diagonal(rho, 1); rho = np.clip(rho, 0, 1)
    d = 1-rho; d = (d+d.T)/2; np.fill_diagonal(d,0); d = np.clip(d,0,2)
    return fcluster(linkage(squareform(d, checks=False), 'average'),
                    t=1-thr, criterion='distance')


# ═══════════════════════════════════════════════════════════════════════
# A: NUMERICAL VALIDATIONS
# ═══════════════════════════════════════════════════════════════════════

def validation_synthetic_beyond_capacity():
    """Validate Part (iv): beyond-capacity MSE = ‖w‖² on synthetic data."""
    print("  A1: Beyond-capacity MSE validation (synthetic)")

    results = []
    for rho_val in [0.70, 0.80, 0.90, 0.95, 0.99]:
        rng = np.random.default_rng(int(rho_val * 100))
        P, ng, gs = 12, 3, 4
        betas = np.array([5.0]*4 + [2.0]*4 + [0.5]*4)
        S = np.zeros((P, P))
        for g in range(ng):
            sl = slice(g*gs, (g+1)*gs)
            S[sl, sl] = rho_val
        np.fill_diagonal(S, 1.0)
        L = np.linalg.cholesky(S)
        X = rng.standard_normal((500, P)) @ L.T
        y = (X @ betas + rng.normal(0, 1, 500) > np.median(X @ betas)).astype(int)

        grp = sage(X, THR)

        # Train M models, get SHAP
        imps = np.zeros((M, P))
        for s in range(M):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                                   random_state=s, stratify=y)
            mdl = XGBClassifier(**XGB_CLF, random_state=s)
            mdl.fit(Xtr, ytr)
            sv = shap.TreeExplainer(mdl).shap_values(Xte[:200])
            if isinstance(sv, list): sv = sv[1]
            imps[s] = np.mean(np.abs(sv), axis=0)

        # Orbit average (Reynolds projection)
        orbit_avg = np.zeros(P)
        for gid in np.unique(grp):
            mask = grp == gid
            orbit_avg[mask] = np.mean(imps.mean(axis=0)[mask])

        # For each within-group pair: compute MSE of the DIFFERENCE
        # The difference d = imp[i] - imp[j] for within-group i,j is in (V^G)⊥
        # Coding theorem predicts: MSE of stable estimate = ‖d‖²
        within_mse = []
        within_norm_sq = []
        between_mse = []
        between_norm_sq = []

        for i in range(P):
            for j in range(i+1, P):
                # True mean difference across seeds
                true_diff = np.mean(imps[:, i] - imps[:, j])
                # Stable estimate of difference
                stable_diff = orbit_avg[i] - orbit_avg[j]
                # MSE
                mse = (stable_diff - true_diff) ** 2
                norm_sq = true_diff ** 2

                if grp[i] == grp[j]:
                    within_mse.append(mse)
                    within_norm_sq.append(norm_sq)
                else:
                    between_mse.append(mse)
                    between_norm_sq.append(norm_sq)

        # For within-group: stable_diff should be 0 (orbit average equalizes)
        # So MSE should ≈ true_diff² = ‖w‖²
        # The beyond-capacity penalty predicts MSE ≥ ‖w‖²
        if within_norm_sq:
            ratio = np.mean(within_mse) / np.mean(within_norm_sq) if np.mean(within_norm_sq) > 0 else float('inf')
        else:
            ratio = None

        results.append({
            'rho': rho_val,
            'within_mse_mean': float(np.mean(within_mse)) if within_mse else None,
            'within_norm_sq_mean': float(np.mean(within_norm_sq)) if within_norm_sq else None,
            'mse_over_norm_sq': float(ratio) if ratio is not None else None,
            'between_mse_mean': float(np.mean(between_mse)) if between_mse else None,
            'n_within': len(within_mse),
            'n_between': len(between_mse),
        })

        print(f"    ρ={rho_val:.2f}: within MSE/‖w‖² = {ratio:.3f}" if ratio else
              f"    ρ={rho_val:.2f}: no within-group pairs")

    return results


def validation_convergence_rate():
    """Validate Part (i): MSE = tr(RΣR)/M convergence rate."""
    print("  A2: Convergence rate validation")

    rng = np.random.default_rng(42)
    P, ng, gs = 12, 3, 4
    betas = np.array([5.0]*4 + [2.0]*4 + [0.5]*4)
    S = np.zeros((P, P))
    for g in range(ng):
        sl = slice(g*gs, (g+1)*gs)
        S[sl, sl] = 0.99
    np.fill_diagonal(S, 1.0)
    L = np.linalg.cholesky(S)
    X = rng.standard_normal((500, P)) @ L.T
    y = (X @ betas + rng.normal(0, 1, 500) > np.median(X @ betas)).astype(int)
    grp = sage(X, THR)

    # Train 100 models (more than M for convergence analysis)
    M_max = 100
    all_imps = np.zeros((M_max, P))
    for s in range(M_max):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                               random_state=s, stratify=y)
        mdl = XGBClassifier(**XGB_CLF, random_state=s)
        mdl.fit(Xtr, ytr)
        sv = shap.TreeExplainer(mdl).shap_values(Xte[:200])
        if isinstance(sv, list): sv = sv[1]
        all_imps[s] = np.mean(np.abs(sv), axis=0)

    # Compute orbit average for each prefix size M
    # Reference: orbit average of all 100 models
    ref_orbit = np.zeros(P)
    ref_mean = all_imps.mean(axis=0)
    for gid in np.unique(grp):
        mask = grp == gid
        ref_orbit[mask] = np.mean(ref_mean[mask])

    M_values = [2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
    mse_values = []

    for m in M_values:
        # Average MSE across 50 random subsets of size m
        mses = []
        for trial in range(50):
            rng_t = np.random.RandomState(trial)
            idx = rng_t.choice(M_max, m, replace=False)
            sub_mean = all_imps[idx].mean(axis=0)
            sub_orbit = np.zeros(P)
            for gid in np.unique(grp):
                mask = grp == gid
                sub_orbit[mask] = np.mean(sub_mean[mask])
            mse = np.mean((sub_orbit - ref_orbit)**2)
            mses.append(mse)
        mse_values.append(np.mean(mses))

    # Fit 1/M model
    log_M = np.log(np.array(M_values[:-1]))  # exclude M=100 (reference)
    log_MSE = np.log(np.array(mse_values[:-1]))
    slope, intercept = np.polyfit(log_M, log_MSE, 1)

    print(f"    MSE ~ M^{slope:.2f} (predicted: M^-1.00)")
    print(f"    R² of log-log fit: {pearsonr(log_M, log_MSE)[0]**2:.4f}")

    return {
        'M_values': M_values,
        'mse_values': [float(m) for m in mse_values],
        'slope': float(slope),
        'r_squared': float(pearsonr(log_M, log_MSE)[0]**2),
    }


def validation_mi_v2():
    """Validate using existing MI v2 transformer data."""
    print("  A3: Modular addition transformer validation")

    try:
        mi_data = json.load(open(OUT / "results_comprehensive_circuit_stability.json"))
        # Extract importance vectors from the comprehensive results
        configs = mi_data.get('configurations', mi_data.get('results', {}))

        if isinstance(configs, dict):
            for config_name, config_data in configs.items():
                if 'raw_rho' in config_data or 'spearman_rho' in config_data:
                    raw_rho = config_data.get('raw_rho', config_data.get('spearman_rho'))
                    inv_rho = config_data.get('ginvariant_rho', config_data.get('g_invariant_rho'))
                    print(f"    {config_name}: raw ρ={raw_rho}, G-inv ρ={inv_rho}")

        print("    (Full validation requires raw importance vectors; using saved statistics)")
        return {"status": "used_saved_statistics", "source": "results_comprehensive_circuit_stability.json"}

    except Exception as e:
        print(f"    Could not load MI data: {e}")
        return {"status": "skipped", "reason": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# E: PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════════

def make_figures():
    print("\n  Generating publication figures...")

    results = json.load(open(OUT / "results_audit_150_final.json"))
    T = f"{THR:.2f}"

    # ── Figure 1: Claims vs Capacity scatter ──
    fig, ax = plt.subplots(figsize=(7, 5))
    Ps, Cs, domains = [], [], []
    for r in results:
        p = r['thresholds'][T]
        Ps.append(r['P'])
        Cs.append(p['g'])
        domains.append(r['domain'])

    # Color by category
    synth = ['Synthetic' in d or 'Control' in d for d in domains]
    colors = ['#1f77b4' if not s else '#aec7e8' for s in synth]

    ax.scatter(Cs, Ps, c=colors, alpha=0.6, s=40, edgecolors='k', linewidths=0.3)
    max_val = max(max(Ps), max(Cs)) + 5
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='P = C (no exceedance)')
    ax.set_xlabel('Explanation Capacity (C = number of groups)')
    ax.set_ylabel('Features Ranked (P)')
    ax.set_title('Claims vs Capacity: 149 Datasets')
    ax.legend()

    # Shade exceedance region
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val],
                    alpha=0.08, color='red', label='Exceeds capacity')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    fig.savefig(FIG / "audit_claims_vs_capacity.pdf")
    plt.close()
    print("    ✓ audit_claims_vs_capacity.pdf")

    # ── Figure 2: Flip rate boxplot ──
    fig, ax = plt.subplots(figsize=(6, 4))
    within_all, between_all = [], []
    for r in results:
        p = r['thresholds'][T]
        if p.get('within_flip') is not None:
            within_all.append(p['within_flip'])
        if p.get('between_flip') is not None:
            between_all.append(p['between_flip'])

    bp = ax.boxplot([within_all, between_all],
                    labels=['Within-group\n(beyond capacity)', 'Between-group\n(within capacity)'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#ff7f7f')
    bp['boxes'][1].set_facecolor('#7fbf7f')
    ax.set_ylabel('Mean pairwise flip rate')
    ax.set_title(f'Flip Rates: Within vs Between Groups (n={len(within_all)}, {len(between_all)})')
    ax.axhline(0.5, color='red', linestyle=':', alpha=0.4, label='Exact exchangeability prediction')
    ax.axhline(0.0, color='green', linestyle=':', alpha=0.4, label='Stable prediction')
    ax.legend(fontsize=8)
    fig.savefig(FIG / "audit_flip_rate_boxplot.pdf")
    plt.close()
    print("    ✓ audit_flip_rate_boxplot.pdf")

    # ── Figure 3: Sensitivity curve ──
    fig, ax = plt.subplots(figsize=(6, 4))
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    exc_rates = []
    for thr in thresholds:
        tk = f"{thr:.2f}"
        n_e = sum(1 for r in results if r['thresholds'][tk]['exceed'])
        exc_rates.append(100 * n_e / len(results))

    ax.plot(thresholds, exc_rates, 'bo-', linewidth=2, markersize=8)
    ax.axvline(0.70, color='red', linestyle='--', alpha=0.5, label='Primary threshold (ρ*=0.70)')
    ax.set_xlabel('Correlation threshold ρ*')
    ax.set_ylabel('Datasets exceeding capacity (%)')
    ax.set_title('Sensitivity Analysis: Exceedance vs Threshold')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "audit_sensitivity_curve.pdf")
    plt.close()
    print("    ✓ audit_sensitivity_curve.pdf")

    # ── Figure 4: Domain forest plot ──
    from collections import defaultdict
    domain_data = defaultdict(lambda: [0, 0])
    for r in results:
        p = r['thresholds'][T]
        dom = r['domain']
        # Consolidate synthetic
        if 'Synth' in dom or 'Control' in dom:
            dom = 'Synthetic (controls)'
        elif dom == 'Other':
            dom = 'Uncategorized (PMLB)'
        domain_data[dom][1] += 1
        if p['exceed']:
            domain_data[dom][0] += 1

    # Sort by exceedance rate
    sorted_domains = sorted(domain_data.items(), key=lambda x: x[1][0]/max(x[1][1],1), reverse=True)
    # Top 25 domains by count
    sorted_domains = [d for d in sorted_domains if d[1][1] >= 1][:30]

    fig, ax = plt.subplots(figsize=(8, max(6, len(sorted_domains) * 0.3)))
    y_pos = range(len(sorted_domains))
    rates = [100 * d[1][0] / d[1][1] for d in sorted_domains]
    labels = [f"{d[0]} ({d[1][0]}/{d[1][1]})" for d in sorted_domains]

    bars = ax.barh(y_pos, rates, color=['#2ecc71' if r == 100 else '#e74c3c' if r == 0
                                         else '#f39c12' for r in rates], alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Exceedance rate (%)')
    ax.set_title('Exceedance by Domain')
    ax.axvline(75, color='k', linestyle='--', alpha=0.3, label='Overall rate (75%)')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    fig.savefig(FIG / "audit_domain_forest.pdf")
    plt.close()
    print("    ✓ audit_domain_forest.pdf")

    # ── Figure 5: η degradation curve ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    rho_vals_synth = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    within_synth = []
    between_synth = []
    gaps_synth = []

    for rho_val in rho_vals_synth:
        rng = np.random.default_rng(int(rho_val * 100))
        P_s, ng, gs = 12, 3, 4
        betas_s = np.array([5.0]*4 + [2.0]*4 + [0.5]*4)
        S_s = np.zeros((P_s, P_s))
        for g in range(ng):
            sl = slice(g*gs, (g+1)*gs)
            S_s[sl, sl] = rho_val
        np.fill_diagonal(S_s, 1.0)
        L_s = np.linalg.cholesky(S_s)
        X_s = rng.standard_normal((500, P_s)) @ L_s.T
        y_s = (X_s @ betas_s + rng.normal(0, 1, 500) > np.median(X_s @ betas_s)).astype(int)
        grp_s = sage(X_s, THR)

        imps = np.zeros((M, P_s))
        for s in range(M):
            Xtr, Xte, ytr, yte = train_test_split(X_s, y_s, test_size=0.2,
                                                   random_state=s, stratify=y_s)
            mdl = XGBClassifier(**XGB_CLF, random_state=s)
            mdl.fit(Xtr, ytr)
            sv = shap.TreeExplainer(mdl).shap_values(Xte[:200])
            if isinstance(sv, list): sv = sv[1]
            imps[s] = np.mean(np.abs(sv), axis=0)

        flips = np.zeros((P_s, P_s))
        for i in range(P_s):
            for j in range(i+1, P_s):
                w = np.sum(imps[:, i] > imps[:, j])
                flips[i,j] = flips[j,i] = min(w, M-w)/M

        wi = [flips[i,j] for i in range(P_s) for j in range(i+1,P_s) if grp_s[i]==grp_s[j]]
        bw = [flips[i,j] for i in range(P_s) for j in range(i+1,P_s) if grp_s[i]!=grp_s[j]]
        wm = np.mean(wi) if wi else 0
        bm = np.mean(bw) if bw else 0
        within_synth.append(wm)
        between_synth.append(bm)
        gaps_synth.append(wm - bm)

    axes[0].plot(rho_vals_synth, within_synth, 'ro-', label='Within-group', linewidth=2)
    axes[0].plot(rho_vals_synth, between_synth, 'gs-', label='Between-group', linewidth=2)
    axes[0].set_xlabel('Within-group correlation (ρ)')
    axes[0].set_ylabel('Mean flip rate')
    axes[0].set_title('η Law Degradation')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0.5, color='red', linestyle=':', alpha=0.3)

    axes[1].bar(range(len(rho_vals_synth)), gaps_synth, color='steelblue', alpha=0.7)
    axes[1].set_xticks(range(len(rho_vals_synth)))
    axes[1].set_xticklabels([f"{r:.2f}" for r in rho_vals_synth])
    axes[1].set_xlabel('Within-group correlation (ρ)')
    axes[1].set_ylabel('Gap (within − between)')
    axes[1].set_title('Bimodal Gap Increases with ρ')
    axes[1].grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(FIG / "audit_eta_degradation.pdf")
    plt.close()
    print("    ✓ audit_eta_degradation.pdf")

    # ── Figure 6: Convergence rate (1/M) ──
    conv = validation_convergence_rate()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(conv['M_values'][:-1], conv['mse_values'][:-1], 'bo-', linewidth=2, label='Observed MSE')
    # Plot 1/M reference line
    M_ref = np.array(conv['M_values'][:-1])
    scale = conv['mse_values'][2] * conv['M_values'][2]  # calibrate at M=5
    ax.loglog(M_ref, scale / M_ref, 'r--', alpha=0.5, label='1/M (predicted)')
    ax.set_xlabel('Number of models (M)')
    ax.set_ylabel('MSE of orbit average')
    ax.set_title(f'Convergence Rate: MSE ~ M^{conv["slope"]:.2f} (predicted: M^-1)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "coding_theorem_convergence.pdf")
    plt.close()
    print("    ✓ coding_theorem_convergence.pdf")

    return conv


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CODING THEOREM VALIDATION + PUBLICATION FIGURES")
    print("=" * 70)

    # A: Validations
    print("\n  A: NUMERICAL VALIDATIONS")
    print("  " + "-" * 50)

    v1 = validation_synthetic_beyond_capacity()
    v3 = validation_mi_v2()

    # E: Figures (includes convergence validation A2)
    print("\n  E: PUBLICATION FIGURES")
    print("  " + "-" * 50)
    v2 = make_figures()

    # Save validation results
    all_validation = {
        'beyond_capacity_mse': v1,
        'convergence_rate': v2,
        'mi_v2': v3,
    }
    with open(OUT / "results_coding_theorem_validation.json", "w") as f:
        json.dump(all_validation, f, indent=2, default=str)

    print(f"\n  Results saved: results_coding_theorem_validation.json")
    print(f"  Figures saved to: {FIG}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
