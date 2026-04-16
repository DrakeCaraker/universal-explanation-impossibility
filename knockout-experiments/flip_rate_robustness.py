#!/usr/bin/env python3
"""
Flip Rate Robustness: Phi(-SNR) under non-Gaussian distributions.

For each distribution family and each SNR in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]:
1. Generate 100,000 pairs of attribution differences from distribution
2. Compute actual flip rate (fraction where sign flips)
3. Compare to Phi(-SNR) prediction
4. Report absolute error, relative error

Distributions:
- Gaussian (N(0,1)) -- calibration check
- t-distribution (df=3, 5, 10, 30) -- heavy tails
- Log-normal (sigma=0.5, 1.0) -- skewness
- Mixture of 2 Gaussians (0.5*N(-1,1) + 0.5*N(1,1)) -- multimodality
- Uniform(-sqrt(3), sqrt(3)) -- light tails (same variance as N(0,1))

Also compute Berry-Esseen bound for n = 50, 100, 500, 1000, 10000
at each SNR. Report as % of base rate.

All with fixed seed=42 for reproducibility.
"""

import json
import numpy as np
from scipy.stats import norm, t as t_dist
from pathlib import Path

OUT_DIR = Path(__file__).parent

np.random.seed(42)

N_SAMPLES = 100_000
SNR_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
BERRY_ESSEEN_N = [50, 100, 500, 1000, 10000]
BERRY_ESSEEN_CONSTANT = 0.4748  # Shevtsov's improvement of the B-E constant


# ---- Distribution families ----
# Each returns N_SAMPLES draws with mean=0 and variance=1 (standardized)
# Then we shift by Delta to get mean=Delta, so flip = P(X < 0) where X ~ F(Delta, 1)

def gen_gaussian(n):
    return np.random.normal(0, 1, n)

def gen_t(n, df):
    """t-distribution scaled to unit variance. Var(t_df) = df/(df-2) for df>2."""
    raw = t_dist.rvs(df, size=n, random_state=None)
    if df > 2:
        raw = raw / np.sqrt(df / (df - 2))
    return raw

def gen_lognormal(n, sigma_param):
    """Log-normal shifted and scaled to mean=0, variance=1.
    LN(mu_ln, sigma_ln) has mean=exp(mu_ln + sigma_ln^2/2), var=(exp(sigma_ln^2)-1)*exp(2*mu_ln+sigma_ln^2).
    We generate LN(0, sigma_param), then standardize."""
    raw = np.random.lognormal(0, sigma_param, n)
    return (raw - np.mean(raw)) / np.std(raw)

def gen_mixture(n):
    """0.5*N(-1,1) + 0.5*N(1,1). Mean=0, Var=1+1=2 (mixture variance). Standardize."""
    components = np.random.binomial(1, 0.5, n)
    raw = np.where(components, np.random.normal(1, 1, n), np.random.normal(-1, 1, n))
    return (raw - np.mean(raw)) / np.std(raw)

def gen_uniform(n):
    """Uniform(-sqrt(3), sqrt(3)). Mean=0, Var=1."""
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), n)


# Build distribution catalog
distributions = {
    "Gaussian N(0,1)": lambda n: gen_gaussian(n),
    "t(df=3)": lambda n: gen_t(n, 3),
    "t(df=5)": lambda n: gen_t(n, 5),
    "t(df=10)": lambda n: gen_t(n, 10),
    "t(df=30)": lambda n: gen_t(n, 30),
    "LogNormal(sigma=0.5)": lambda n: gen_lognormal(n, 0.5),
    "LogNormal(sigma=1.0)": lambda n: gen_lognormal(n, 1.0),
    "Mixture(0.5*N(-1,1)+0.5*N(1,1))": lambda n: gen_mixture(n),
    "Uniform(-sqrt3, sqrt3)": lambda n: gen_uniform(n),
}


def compute_third_abs_moment(samples):
    """E[|Z|^3] for centered samples."""
    centered = samples - np.mean(samples)
    return np.mean(np.abs(centered) ** 3)


# ---- Main experiment ----
print("=" * 75)
print("EXPERIMENT 3: Flip Rate Robustness to Non-Gaussianity + Berry-Esseen")
print("=" * 75)

results = {
    "experiment": "Flip Rate Robustness to Non-Gaussianity",
    "n_samples": N_SAMPLES,
    "snr_values": SNR_VALUES,
    "distributions": {},
    "berry_esseen": {},
    "summary": {}
}

# Part 1: Empirical flip rates vs Gaussian prediction
print("\n--- Part 1: Empirical flip rates vs Phi(-SNR) ---\n")
print(f"{'Distribution':<40} {'SNR':>5} {'Phi(-SNR)':>10} {'Actual':>10} {'AbsErr':>10} {'RelErr%':>10}")
print("-" * 95)

max_abs_error = 0.0
max_rel_error = 0.0
worst_case = None

for dist_name, gen_fn in distributions.items():
    dist_results = {}

    for snr in SNR_VALUES:
        # Reset seed per (dist, snr) for reproducibility
        np.random.seed(42)

        # Generate noise with mean=0, var=1, then shift by SNR (= Delta/sigma, with sigma=1)
        noise = gen_fn(N_SAMPLES)

        # X = Delta + noise, where Delta = SNR * sigma = SNR * 1 = SNR
        # Flip occurs when X < 0, i.e., noise < -SNR
        X = snr + noise
        actual_flip = np.mean(X < 0)

        # Gaussian prediction
        gaussian_pred = norm.cdf(-snr)

        abs_err = abs(actual_flip - gaussian_pred)
        rel_err = (abs_err / gaussian_pred * 100) if gaussian_pred > 1e-10 else float('inf')

        dist_results[str(snr)] = {
            "gaussian_prediction": round(float(gaussian_pred), 6),
            "actual_flip_rate": round(float(actual_flip), 6),
            "absolute_error": round(float(abs_err), 6),
            "relative_error_pct": round(float(rel_err), 4),
            "noise_mean": round(float(np.mean(noise)), 6),
            "noise_std": round(float(np.std(noise)), 6),
            "noise_skew": round(float(np.mean((noise - np.mean(noise))**3) / np.std(noise)**3), 4),
            "noise_kurtosis_excess": round(float(np.mean((noise - np.mean(noise))**4) / np.std(noise)**4 - 3), 4),
        }

        if abs_err > max_abs_error:
            max_abs_error = abs_err
            worst_case = (dist_name, snr, gaussian_pred, actual_flip, abs_err, rel_err)

        if rel_err > max_rel_error and gaussian_pred > 0.001:
            max_rel_error = rel_err

        print(f"{dist_name:<40} {snr:>5.2f} {gaussian_pred:>10.6f} {actual_flip:>10.6f} {abs_err:>10.6f} {rel_err:>10.2f}")

    results["distributions"][dist_name] = dist_results

print()

# Part 2: Berry-Esseen bounds
print("\n--- Part 2: Berry-Esseen Bounds ---\n")
print(f"{'Distribution':<40} {'E[|Z|^3]':>10} ", end="")
for n_be in BERRY_ESSEEN_N:
    print(f"{'n='+str(n_be):>12}", end="")
print()
print("-" * (52 + 12 * len(BERRY_ESSEEN_N)))

for dist_name, gen_fn in distributions.items():
    np.random.seed(42)
    samples = gen_fn(N_SAMPLES)
    sigma = np.std(samples)
    third_moment = compute_third_abs_moment(samples)

    be_results = {
        "third_absolute_moment": round(float(third_moment), 6),
        "sigma": round(float(sigma), 6),
    }

    print(f"{dist_name:<40} {third_moment:>10.4f} ", end="")

    for n_be in BERRY_ESSEEN_N:
        # Berry-Esseen bound: |F_n(x) - Phi(x)| <= C * rho / (sigma^3 * sqrt(n))
        # where rho = E[|Z|^3], sigma = std(Z), C = 0.4748
        be_bound = BERRY_ESSEEN_CONSTANT * third_moment / (sigma**3 * np.sqrt(n_be))
        be_results[f"n={n_be}"] = {
            "bound": round(float(be_bound), 6),
        }

        # Also compute as % of base rate for each SNR
        snr_pcts = {}
        for snr in SNR_VALUES:
            base_rate = norm.cdf(-snr)
            if base_rate > 1e-10:
                pct_of_base = be_bound / base_rate * 100
            else:
                pct_of_base = float('inf')
            snr_pcts[str(snr)] = round(float(pct_of_base), 2)
        be_results[f"n={n_be}"]["pct_of_base_by_snr"] = snr_pcts

        print(f"{be_bound:>12.6f}", end="")

    print()
    results["berry_esseen"][dist_name] = be_results

# Part 3: Summary statistics
print("\n\n--- Part 3: Summary ---\n")

# Collect all absolute errors
all_abs_errors = []
all_rel_errors = []
non_gaussian_abs_errors = []
non_gaussian_rel_errors = []

for dist_name, dist_data in results["distributions"].items():
    for snr_key, snr_data in dist_data.items():
        all_abs_errors.append(snr_data["absolute_error"])
        if snr_data["gaussian_prediction"] > 0.001:
            all_rel_errors.append(snr_data["relative_error_pct"])
        if "Gaussian" not in dist_name:
            non_gaussian_abs_errors.append(snr_data["absolute_error"])
            if snr_data["gaussian_prediction"] > 0.001:
                non_gaussian_rel_errors.append(snr_data["relative_error_pct"])

summary = {
    "all_distributions": {
        "n_conditions": len(all_abs_errors),
        "mean_absolute_error": round(float(np.mean(all_abs_errors)), 6),
        "max_absolute_error": round(float(np.max(all_abs_errors)), 6),
        "median_absolute_error": round(float(np.median(all_abs_errors)), 6),
        "mean_relative_error_pct": round(float(np.mean(all_rel_errors)), 2),
        "max_relative_error_pct": round(float(np.max(all_rel_errors)), 2),
        "median_relative_error_pct": round(float(np.median(all_rel_errors)), 2),
    },
    "non_gaussian_only": {
        "n_conditions": len(non_gaussian_abs_errors),
        "mean_absolute_error": round(float(np.mean(non_gaussian_abs_errors)), 6),
        "max_absolute_error": round(float(np.max(non_gaussian_abs_errors)), 6),
        "median_absolute_error": round(float(np.median(non_gaussian_abs_errors)), 6),
        "mean_relative_error_pct": round(float(np.mean(non_gaussian_rel_errors)), 2),
        "max_relative_error_pct": round(float(np.max(non_gaussian_rel_errors)), 2),
        "median_relative_error_pct": round(float(np.median(non_gaussian_rel_errors)), 2),
    },
    "worst_case": {
        "distribution": worst_case[0],
        "snr": worst_case[1],
        "gaussian_prediction": round(float(worst_case[2]), 6),
        "actual_flip_rate": round(float(worst_case[3]), 6),
        "absolute_error": round(float(worst_case[4]), 6),
        "relative_error_pct": round(float(worst_case[5]), 2),
    },
    "berry_esseen_n1000_snr1": {},
}

# Berry-Esseen at n=1000, SNR=1.0 for each distribution
for dist_name in results["berry_esseen"]:
    be_data = results["berry_esseen"][dist_name]
    bound_at_1000 = be_data["n=1000"]["bound"]
    base_rate = norm.cdf(-1.0)
    summary["berry_esseen_n1000_snr1"][dist_name] = {
        "bound": bound_at_1000,
        "base_rate": round(float(base_rate), 6),
        "bound_as_pct_of_base": round(float(bound_at_1000 / base_rate * 100), 2),
    }

results["summary"] = summary

# Verdict
print(f"Total conditions tested: {len(all_abs_errors)}")
print(f"Mean absolute error (all): {np.mean(all_abs_errors):.6f}")
print(f"Max absolute error (all):  {np.max(all_abs_errors):.6f}")
print(f"Mean absolute error (non-Gaussian): {np.mean(non_gaussian_abs_errors):.6f}")
print(f"Max absolute error (non-Gaussian):  {np.max(non_gaussian_abs_errors):.6f}")
print(f"\nMean relative error (all): {np.mean(all_rel_errors):.2f}%")
print(f"Max relative error (all):  {np.max(all_rel_errors):.2f}%")
print(f"Mean relative error (non-Gaussian): {np.mean(non_gaussian_rel_errors):.2f}%")
print(f"Max relative error (non-Gaussian):  {np.max(non_gaussian_rel_errors):.2f}%")
print(f"\nWorst case: {worst_case[0]} at SNR={worst_case[1]}")
print(f"  Gaussian prediction: {worst_case[2]:.6f}")
print(f"  Actual flip rate:    {worst_case[3]:.6f}")
print(f"  Absolute error:      {worst_case[4]:.6f}")
print(f"  Relative error:      {worst_case[5]:.2f}%")

# Berry-Esseen interpretation
print("\nBerry-Esseen bounds at n=1000, SNR=1.0:")
for dist_name, be_info in summary["berry_esseen_n1000_snr1"].items():
    print(f"  {dist_name:<40} bound={be_info['bound']:.6f} ({be_info['bound_as_pct_of_base']:.1f}% of base rate)")

# Final verdict
threshold_abs = 0.05
threshold_rel = 30.0  # %
all_pass_abs = np.max(non_gaussian_abs_errors) < threshold_abs
all_pass_rel = np.max(non_gaussian_rel_errors) < threshold_rel

if all_pass_abs and all_pass_rel:
    verdict = f"ROBUST: Max non-Gaussian absolute error {np.max(non_gaussian_abs_errors):.4f} < {threshold_abs}, max relative error {np.max(non_gaussian_rel_errors):.1f}% < {threshold_rel}%"
elif all_pass_abs:
    verdict = f"PARTIALLY ROBUST: Abs error OK ({np.max(non_gaussian_abs_errors):.4f} < {threshold_abs}), but relative error high ({np.max(non_gaussian_rel_errors):.1f}% >= {threshold_rel}%)"
else:
    verdict = f"NOT ROBUST: Max non-Gaussian abs error {np.max(non_gaussian_abs_errors):.4f} >= {threshold_abs}"

results["verdict"] = verdict
print(f"\nVERDICT: {verdict}")

# Save results
out_path = OUT_DIR / "results_flip_rate_robustness.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
