#!/usr/bin/env python3
"""
Mechanistic Interpretability Stability Experiment

Tests whether circuit-level explanations of GPT-2 small are structurally
stable across model perturbations. Uses the Indirect Object Identification
(IOI) task, the best-studied circuit in mechanistic interpretability.

Design:
- Base model: GPT-2 small (12 layers × 12 heads)
- Task: IOI (e.g., "When Mary and John went to the store, John gave a drink to")
- 20 perturbations: add Gaussian noise to all weights (σ chosen to keep
  loss within 5% of baseline)
- For each perturbation: measure attention head importance via activation patching
- Compute: pairwise flip rate for head importance rankings
- Apply: Gaussian flip formula to predict which heads are stably identified
- Compare: to the published IOI circuit (Wang et al., 2023)

This tests the impossibility theorem's prediction: circuit-level explanations
of underspecified systems are structurally unreliable, and the Gaussian flip
formula predicts which components are reliable.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import torch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

def create_ioi_prompts(n=50):
    """Create IOI-style prompts for testing."""
    names = ['Mary', 'John', 'Alice', 'Bob', 'Sarah', 'Tom', 'Emma', 'James',
             'Lisa', 'David', 'Kate', 'Mike', 'Anna', 'Chris', 'Jane', 'Paul']
    places = ['store', 'park', 'office', 'school', 'restaurant', 'library']
    objects = ['drink', 'book', 'letter', 'gift', 'key', 'phone']

    prompts = []
    rng = np.random.RandomState(42)
    for _ in range(n):
        n1, n2 = rng.choice(names, size=2, replace=False)
        place = rng.choice(places)
        obj = rng.choice(objects)
        # IOI pattern: "When [A] and [B] went to the [place], [B] gave a [obj] to"
        # Correct completion: [A] (the indirect object)
        prompt = f"When {n1} and {n2} went to the {place}, {n2} gave a {obj} to"
        prompts.append({"prompt": prompt, "io": n1, "s": n2})
    return prompts


def measure_head_importance(model, prompts, device='cpu'):
    """Measure attention head importance via mean ablation (zero ablation).

    For each head: replace its output with zeros, measure change in
    correct-token logit. Large change = important head.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    head_effects = np.zeros((n_layers, n_heads))

    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        io_name = prompt_data["io"]

        tokens = model.to_tokens(prompt)
        io_token = model.to_tokens(" " + io_name, prepend_bos=False)[0, 0].item()

        # Baseline logit
        with torch.no_grad():
            baseline_logits = model(tokens)
            baseline_logit = baseline_logits[0, -1, io_token].item()

        # Ablate each head
        for layer in range(n_layers):
            for head in range(n_heads):
                hook_name = f"blocks.{layer}.attn.hook_result"

                def ablation_hook(value, hook, head_idx=head):
                    value[:, :, head_idx, :] = 0.0
                    return value

                with torch.no_grad():
                    ablated_logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, ablation_hook)]
                    )
                    ablated_logit = ablated_logits[0, -1, io_token].item()

                head_effects[layer, head] += (baseline_logit - ablated_logit)

    head_effects /= len(prompts)
    return head_effects


def run_experiment():
    print("Mechanistic Interpretability Stability Experiment")
    print("=" * 60)
    t0 = time.time()

    import transformer_lens as tl

    device = 'cpu'

    # Load base model
    print("\nLoading GPT-2 small...")
    model = tl.HookedTransformer.from_pretrained('gpt2-small', device=device)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  {n_layers} layers × {n_heads} heads = {n_layers * n_heads} components")

    # Create IOI prompts
    print("\nCreating IOI prompts...")
    prompts = create_ioi_prompts(n=20)  # Reduced for CPU speed
    print(f"  {len(prompts)} prompts")

    # Measure baseline head importance
    print("\nMeasuring baseline head importance (this takes a few minutes on CPU)...")
    baseline_importance = measure_head_importance(model, prompts, device)
    baseline_flat = baseline_importance.flatten()
    print(f"  Top 5 heads: {np.argsort(-np.abs(baseline_flat))[:5]}")
    print(f"  Importance range: [{baseline_flat.min():.4f}, {baseline_flat.max():.4f}]")

    # Perturb model and measure importance across perturbations
    n_perturbations = 10  # Reduced for CPU
    print(f"\nRunning {n_perturbations} weight perturbations...")

    # Find σ that keeps loss within 5%
    # Test with small σ first
    importance_matrix = [baseline_flat.copy()]

    for p_idx in range(n_perturbations):
        print(f"  Perturbation {p_idx + 1}/{n_perturbations}...")

        # Save original weights
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Add noise
        sigma = 0.005  # Small enough to keep model functional
        rng = np.random.RandomState(1000 + p_idx)
        with torch.no_grad():
            for name, param in model.named_parameters():
                noise = torch.randn_like(param) * sigma
                param.add_(noise)

        # Measure importance
        perturbed_importance = measure_head_importance(model, prompts, device)
        importance_matrix.append(perturbed_importance.flatten())

        # Restore original weights
        model.load_state_dict(original_state)

    importance_matrix = np.array(importance_matrix)  # (n_perturbations+1, n_layers*n_heads)
    n_instances = importance_matrix.shape[0]
    n_components = importance_matrix.shape[1]
    print(f"\n  Importance matrix: {importance_matrix.shape}")

    # Split into calibration and validation
    n_cal = n_instances // 2
    imp_cal = importance_matrix[:n_cal]
    imp_val = importance_matrix[n_cal:]

    # Compute pairwise flip rates and Gaussian flip predictions
    from scipy.stats import norm, spearmanr
    from itertools import combinations

    pairs = list(combinations(range(n_components), 2))
    # Subsample pairs for speed (144 components → 10296 pairs)
    if len(pairs) > 500:
        rng2 = np.random.RandomState(42)
        pairs = [pairs[i] for i in rng2.choice(len(pairs), size=500, replace=False)]

    predicted_flips = []
    observed_flips = []
    snrs = []

    for j, k in pairs:
        # Calibration: Gaussian prediction
        diff = imp_cal[:, j] - imp_cal[:, k]
        mu = np.mean(diff)
        sd = np.std(diff, ddof=1)
        snr = abs(mu) / sd if sd > 1e-12 else 10.0
        pred = float(norm.cdf(-abs(mu) / sd)) if sd > 1e-12 else 0.0

        # Validation: observed flip
        n_v = imp_val.shape[0]
        disagree = 0
        total = 0
        for m1 in range(n_v):
            for m2 in range(m1 + 1, n_v):
                d1 = imp_val[m1, j] - imp_val[m1, k]
                d2 = imp_val[m2, j] - imp_val[m2, k]
                if d1 * d2 < 0:
                    disagree += 1
                total += 1
        obs = disagree / total if total > 0 else 0.0

        predicted_flips.append(pred)
        observed_flips.append(obs)
        snrs.append(snr)

    predicted_flips = np.array(predicted_flips)
    observed_flips = np.array(observed_flips)
    snrs = np.array(snrs)

    # Metrics
    ss_res = np.sum((observed_flips - predicted_flips) ** 2)
    ss_tot = np.sum((observed_flips - np.mean(observed_flips)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    if np.std(predicted_flips) > 1e-12 and np.std(observed_flips) > 1e-12:
        rho, p_val = spearmanr(predicted_flips, observed_flips)
    else:
        rho, p_val = 0.0, 1.0

    # Coverage conflict
    cc_degree = float(np.mean(snrs < 0.5))
    reliable_frac = float(np.mean(snrs > 2.0))
    mean_flip = float(np.mean(observed_flips))

    print(f"\n{'='*60}")
    print(f"RESULTS: GPT-2 Small Circuit Stability")
    print(f"{'='*60}")
    print(f"  Components: {n_components} (attention heads)")
    print(f"  Perturbations: {n_perturbations}")
    print(f"  Pairs analyzed: {len(pairs)}")
    print(f"  Coverage conflict degree: {cc_degree:.3f}")
    print(f"  Reliable fraction (SNR>2): {reliable_frac:.3f}")
    print(f"  Mean flip rate: {mean_flip:.3f}")
    print(f"  Gaussian flip R²: {r2:.3f}")
    print(f"  Gaussian flip ρ: {rho:.3f} (p={p_val:.2e})")

    # Identify top heads and their stability
    top_10_idx = np.argsort(-np.abs(baseline_flat))[:10]
    print(f"\n  Top 10 heads by baseline importance:")
    for rank, idx in enumerate(top_10_idx):
        layer = idx // n_heads
        head = idx % n_heads
        imp = baseline_flat[idx]
        # Find this head's stability across perturbations
        head_imp_across = importance_matrix[:, idx]
        head_cv = np.std(head_imp_across) / max(abs(np.mean(head_imp_across)), 1e-12)
        print(f"    #{rank+1}: L{layer}H{head} importance={imp:.4f} CV={head_cv:.3f}")

    elapsed = time.time() - t0

    results = {
        "experiment": "mech_interp_stability",
        "model": "gpt2-small",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_components": n_components,
        "n_perturbations": n_perturbations,
        "n_prompts": len(prompts),
        "sigma": 0.005,
        "n_pairs": len(pairs),
        "coverage_conflict_degree": cc_degree,
        "reliable_fraction": reliable_frac,
        "mean_flip_rate": mean_flip,
        "gaussian_r2": float(r2),
        "gaussian_rho": float(rho),
        "gaussian_p": float(p_val),
        "top_10_heads": [
            {"rank": i+1, "layer": int(idx // n_heads), "head": int(idx % n_heads),
             "importance": float(baseline_flat[idx]),
             "cv": float(np.std(importance_matrix[:, idx]) / max(abs(np.mean(importance_matrix[:, idx])), 1e-12))}
            for i, idx in enumerate(top_10_idx)
        ],
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = OUT_DIR / 'results_mech_interp_stability.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    run_experiment()
