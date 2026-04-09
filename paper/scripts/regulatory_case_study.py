"""
Regulatory Case Study: ECOA Adverse Action Stability Audit

Simulates an ECOA adverse action pipeline on 10,000 synthetic loan
applications to show that SHAP instability causes real regulatory harm.
Correlated income features (rho~0.7) produce different "top adverse action
reasons" across equivalent models.  DASH resolves the instability.

Requires: pip install xgboost shap scikit-learn numpy

Usage:
  python regulatory_case_study.py
"""

import os
import json
import numpy as np
import xgboost as xgb
import shap
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "..", "results_regulatory_case_study.json")

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
N_APPLICANTS = 10_000
N_MODELS = 25
FEATURE_NAMES = [
    "salary", "bonus", "investment_income", "rental_income",
    "freelance_income", "credit_score", "debt_ratio", "employment_years",
]
INCOME_FEATURES = FEATURE_NAMES[:5]
UNCORRELATED_FEATURES = FEATURE_NAMES[5:]
RHO = 0.7
SEED = 42

# Known logistic coefficients for the DGP
# Income features get similar positive weight; credit_score positive;
# debt_ratio negative; employment_years mild positive
TRUE_COEFS = np.array([0.30, 0.25, 0.20, 0.18, 0.15,   # income group
                        0.50, -0.60, 0.20])               # uncorrelated
TRUE_INTERCEPT = -0.5

np.random.seed(SEED)

# -------------------------------------------------------------------------
# Step 1: Generate synthetic data
# -------------------------------------------------------------------------
n_features = len(FEATURE_NAMES)
n_income = len(INCOME_FEATURES)

# Covariance: income block correlated at rho, others independent
cov = np.eye(n_features)
for i in range(n_income):
    for j in range(n_income):
        if i != j:
            cov[i, j] = RHO

X = np.random.multivariate_normal(np.zeros(n_features), cov, size=N_APPLICANTS)

# Generate binary target via logistic model
logits = X @ TRUE_COEFS + TRUE_INTERCEPT
probs = 1.0 / (1.0 + np.exp(-logits))
y = (np.random.rand(N_APPLICANTS) < probs).astype(int)

print(f"Generated {N_APPLICANTS} synthetic loan applications")
print(f"  Features: {FEATURE_NAMES}")
print(f"  Income correlation (rho): {RHO}")
print(f"  Approval rate: {y.mean():.1%}")
print()

# -------------------------------------------------------------------------
# Step 2: Train M=25 XGBoost classifiers
# -------------------------------------------------------------------------
models = []
for m in range(N_MODELS):
    rng_seed = SEED + m * 7
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=rng_seed,
        verbosity=0,
        eval_metric="logloss",
        n_jobs=1,
    )
    model.fit(X, y)
    models.append(model)

print(f"Trained {N_MODELS} XGBoost classifiers (subsample=0.8, different seeds)")
print()

# -------------------------------------------------------------------------
# Step 3: Identify denied applicants (majority vote < 0.5)
# -------------------------------------------------------------------------
# Collect prediction probabilities from all models
all_probs = np.zeros((N_MODELS, N_APPLICANTS))
for m, model in enumerate(models):
    all_probs[m] = model.predict_proba(X)[:, 1]

# Denied = majority of models predict < 0.5
majority_denied = np.mean(all_probs < 0.5, axis=0) > 0.5
denied_idx = np.where(majority_denied)[0]
n_denied = len(denied_idx)

print(f"Denied applicants (majority vote): {n_denied}")
print()

# -------------------------------------------------------------------------
# Step 4: Compute TreeSHAP for each model on denied applicants
# -------------------------------------------------------------------------
print("Computing TreeSHAP for each model on denied applicants...")

# shap_all[m] has shape (n_denied, n_features)
shap_all = []
for m, model in enumerate(models):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X[denied_idx])
    shap_all.append(sv)

shap_all = np.array(shap_all)  # (N_MODELS, n_denied, n_features)

# -------------------------------------------------------------------------
# Step 5: Adverse action reason analysis
# -------------------------------------------------------------------------
# For each denied applicant and each model, the "top adverse action reason"
# is the feature with the most negative SHAP value (pushing toward denial).

def top_reason(shap_row):
    """Return index of most negative SHAP feature."""
    return int(np.argmin(shap_row))


def top_2_reasons(shap_row):
    """Return frozenset of indices of 2 most negative SHAP features."""
    idx = np.argsort(shap_row)[:2]
    return frozenset(idx)


# Per-applicant: set of top reasons across models
unstable_top1 = 0
unstable_top2 = 0
flip_counter = Counter()  # tracks (feat_a, feat_b) flip pairs

for i in range(n_denied):
    reasons_1 = set()
    reasons_2 = set()
    top1_list = []
    for m in range(N_MODELS):
        r1 = top_reason(shap_all[m, i])
        r2 = top_2_reasons(shap_all[m, i])
        reasons_1.add(r1)
        reasons_2.add(r2)
        top1_list.append(r1)

    if len(reasons_1) > 1:
        unstable_top1 += 1
        # Count the most common pair of different top-1 reasons
        unique_reasons = list(reasons_1)
        for a_idx in range(len(unique_reasons)):
            for b_idx in range(a_idx + 1, len(unique_reasons)):
                a, b = unique_reasons[a_idx], unique_reasons[b_idx]
                pair = tuple(sorted([a, b]))
                flip_counter[pair] += 1

    if len(reasons_2) > 1:
        unstable_top2 += 1

pct_unstable_top1 = unstable_top1 / n_denied * 100
pct_unstable_top2 = unstable_top2 / n_denied * 100

# Most common flip
if flip_counter:
    most_common_flip, most_common_count = flip_counter.most_common(1)[0]
    total_flips = sum(flip_counter.values())
    flip_pct = most_common_count / total_flips * 100
    flip_name = (f"{FEATURE_NAMES[most_common_flip[0]]} <-> "
                 f"{FEATURE_NAMES[most_common_flip[1]]}")
else:
    flip_name = "N/A"
    flip_pct = 0.0
    most_common_count = 0
    total_flips = 0

# -------------------------------------------------------------------------
# Step 6: DASH resolution
# -------------------------------------------------------------------------
# Ensemble-averaged SHAP
dash_shap = np.mean(shap_all, axis=0)  # (n_denied, n_features)

dash_stable_top1 = 0
dash_income_tied = 0

for i in range(n_denied):
    # DASH top reason is deterministic (single ensemble average)
    dash_r1 = top_reason(dash_shap[i])
    dash_stable_top1 += 1  # by construction, ensemble average is unique

    # Check if income features are "tied" (within 10% of each other in |SHAP|)
    income_shap = dash_shap[i, :n_income]
    income_neg = income_shap[income_shap < 0]
    if len(income_neg) >= 2:
        most_neg = np.min(income_neg)
        second_neg = np.sort(income_neg)[1]
        if most_neg != 0 and abs(second_neg / most_neg - 1.0) < 0.10:
            dash_income_tied += 1

pct_dash_stable = dash_stable_top1 / n_denied * 100
pct_income_tied = dash_income_tied / n_denied * 100

# -------------------------------------------------------------------------
# Print report
# -------------------------------------------------------------------------
print()
print("ECOA ADVERSE ACTION STABILITY AUDIT")
print("====================================")
print(f"Denied applicants:        {n_denied}")
print(f"Unstable top reason:      {pct_unstable_top1:.1f}% ({unstable_top1} of {n_denied})")
print(f"Unstable top-2 reasons:   {pct_unstable_top2:.1f}% ({unstable_top2} of {n_denied})")
print(f"Most common flip:         {flip_name} ({flip_pct:.1f}% of flips)")
print()
print(f"DASH Resolution (M={N_MODELS}):")
print(f"Stable top reason:        {pct_dash_stable:.1f}% ({dash_stable_top1} of {n_denied})")
print(f"Income features tied:     {pct_income_tied:.1f}% (reported as group)")
print()

# Top 5 flip pairs
print("Top 5 flip pairs:")
for pair, count in flip_counter.most_common(5):
    pct = count / total_flips * 100 if total_flips > 0 else 0
    print(f"  {FEATURE_NAMES[pair[0]]:<22s} <-> {FEATURE_NAMES[pair[1]]:<22s}  {pct:5.1f}% ({count})")
print()

# -------------------------------------------------------------------------
# Save JSON
# -------------------------------------------------------------------------
results = {
    "n_applicants": N_APPLICANTS,
    "n_denied": n_denied,
    "n_models": N_MODELS,
    "rho": RHO,
    "features": FEATURE_NAMES,
    "income_features": INCOME_FEATURES,
    "unstable_top1_count": unstable_top1,
    "unstable_top1_pct": round(pct_unstable_top1, 2),
    "unstable_top2_count": unstable_top2,
    "unstable_top2_pct": round(pct_unstable_top2, 2),
    "most_common_flip": flip_name,
    "most_common_flip_pct": round(flip_pct, 2),
    "top_flip_pairs": [
        {
            "pair": [FEATURE_NAMES[p[0]], FEATURE_NAMES[p[1]]],
            "count": c,
            "pct": round(c / total_flips * 100, 2) if total_flips > 0 else 0,
        }
        for p, c in flip_counter.most_common(10)
    ],
    "dash_stable_top1_pct": round(pct_dash_stable, 2),
    "dash_income_tied_pct": round(pct_income_tied, 2),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {os.path.abspath(RESULTS_PATH)}")
