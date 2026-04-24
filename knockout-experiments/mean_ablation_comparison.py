#!/usr/bin/env python3
"""
Mean Ablation Comparison

Runs mean ablation (replace component output with its mean activation)
on the cached Config A and B models and compares with weight-zeroing results.
If both methods show the same pattern, the "weight zeroing only" limitation
is closed.

Run: python3 mean_ablation_comparison.py

Requires: cached models in models_configA/ and models_configB/ from
comprehensive_circuit_stability.py
"""

import warnings
warnings.filterwarnings('ignore')

import json, numpy as np, torch, math
from pathlib import Path
from scipy.stats import spearmanr
from itertools import combinations
from comprehensive_circuit_stability import (
    CONFIGS, DEVICE, TinyLM, load_tinystories, measure_importance_custom,
    g_invariant_projection, compute_flip_rates, N_PATCH_EXAMPLES, NpEncoder
)

OUT_DIR = Path(__file__).resolve().parent


def measure_importance_mean_ablation(model, val_data, cfg):
    """Measure importance via mean ablation: replace each component's output
    with its mean activation across the validation set."""
    model.eval()
    nl, nh = cfg['n_layers'], cfg['n_heads']
    n_comp = nl * nh + nl
    bs = cfg['batch_size']

    # Step 1: Collect mean activations per component
    attn_activations = {l: [] for l in range(nl)}
    mlp_activations = {l: [] for l in range(nl)}
    hooks = []

    for l in range(nl):
        def make_attn_hook(layer):
            def hook(mod, inp, out):
                attn_activations[layer].append(out.detach().cpu())
            return hook

        def make_mlp_hook(layer):
            def hook(mod, inp, out):
                mlp_activations[layer].append(out.detach().cpu())
            return hook

        hooks.append(model.blocks[l].attn.register_forward_hook(make_attn_hook(l)))
        hooks.append(model.blocks[l].mlp.register_forward_hook(make_mlp_hook(l)))

    with torch.no_grad():
        for i in range(0, min(len(val_data), N_PATCH_EXAMPLES), bs):
            batch = val_data[i:i + bs].to(DEVICE)
            model(batch[:, :-1], batch[:, 1:])

    for h in hooks:
        h.remove()

    # Compute mean activation (averaged over batch and sequence position)
    mean_attn = {}
    mean_mlp = {}
    for l in range(nl):
        mean_attn[l] = torch.cat(attn_activations[l]).mean(dim=(0, 1))
        mean_mlp[l] = torch.cat(mlp_activations[l]).mean(dim=(0, 1))

    # Step 2: Baseline perplexity
    base_losses = []
    with torch.no_grad():
        for i in range(0, min(len(val_data), N_PATCH_EXAMPLES), bs):
            batch = val_data[i:i + bs].to(DEVICE)
            _, loss = model(batch[:, :-1], batch[:, 1:])
            base_losses.append(loss.item())
    baseline = np.mean(base_losses)

    importance = np.zeros(n_comp)

    # Step 3: Attention mean ablation (layer-level, distributed to heads)
    for l in range(nl):
        def make_replace_hook(mean_val):
            def hook(mod, inp, out):
                return mean_val.to(out.device).expand_as(out)
            return hook

        h = model.blocks[l].attn.register_forward_hook(
            make_replace_hook(mean_attn[l]))
        abl_losses = []
        with torch.no_grad():
            for i in range(0, min(len(val_data), N_PATCH_EXAMPLES), bs):
                batch = val_data[i:i + bs].to(DEVICE)
                _, loss = model(batch[:, :-1], batch[:, 1:])
                abl_losses.append(loss.item())
        h.remove()
        attn_imp = np.mean(abl_losses) - baseline
        for hh in range(nh):
            importance[l * nh + hh] = attn_imp / nh

    # Step 4: MLP mean ablation
    for l in range(nl):
        def make_replace_hook(mean_val):
            def hook(mod, inp, out):
                return mean_val.to(out.device).expand_as(out)
            return hook

        h = model.blocks[l].mlp.register_forward_hook(
            make_replace_hook(mean_mlp[l]))
        abl_losses = []
        with torch.no_grad():
            for i in range(0, min(len(val_data), N_PATCH_EXAMPLES), bs):
                batch = val_data[i:i + bs].to(DEVICE)
                _, loss = model(batch[:, :-1], batch[:, 1:])
                abl_losses.append(loss.item())
        h.remove()
        importance[nl * nh + l] = np.mean(abl_losses) - baseline

    return importance, baseline


def main():
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    print(f"Device: {DEVICE}")
    _, val_data = load_tinystories(tokenizer)

    all_results = {}

    for config_key in ['A', 'B']:
        cfg = CONFIGS[config_key]
        nl, nh = cfg['n_layers'], cfg['n_heads']
        n_comp = nl * nh + nl
        model_dir = OUT_DIR / cfg['model_dir']

        print(f"\n{'=' * 60}")
        print(f"Config {config_key}: {cfg['name']} — Mean Ablation Comparison")
        print(f"{'=' * 60}")

        all_imp_zero = []
        all_imp_mean = []

        for seed in range(10):
            model = TinyLM(vocab_size, nl, nh, cfg['d_model']).to(DEVICE)
            model.load_state_dict(torch.load(
                model_dir / f'model_seed{seed}.pt',
                map_location=DEVICE, weights_only=True))

            # Weight zeroing (existing method)
            imp_zero, _ = measure_importance_custom(model, val_data, cfg)
            all_imp_zero.append(imp_zero)

            # Mean ablation (new method)
            imp_mean, _ = measure_importance_mean_ablation(model, val_data, cfg)
            all_imp_mean.append(imp_mean)

            print(f"  Seed {seed}: done")

            del model
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

        all_imp_zero = np.array(all_imp_zero)
        all_imp_mean = np.array(all_imp_mean)

        # Compare methods
        zero_rhos = [spearmanr(all_imp_zero[i], all_imp_zero[j])[0]
                     for i, j in combinations(range(10), 2)]
        mean_rhos = [spearmanr(all_imp_mean[i], all_imp_mean[j])[0]
                     for i, j in combinations(range(10), 2)]

        # Cross-method per model (same model, different method)
        cross = [spearmanr(all_imp_zero[i], all_imp_mean[i])[0]
                 for i in range(10)]

        # G-invariant on mean ablation
        proj_mean = g_invariant_projection(all_imp_mean, nl, nh)
        ginv_mean = [spearmanr(proj_mean[i], proj_mean[j])[0]
                     for i, j in combinations(range(10), 2)]

        # Flip rates on mean ablation
        within_m, between_m, _ = compute_flip_rates(all_imp_mean, nl, nh)

        print(f"\n  Results:")
        print(f"  {'Method':<25} {'Full ρ':>8} {'G-inv ρ':>10} {'W-flip':>8} {'B-flip':>8}")
        print(f"  {'-'*60}")
        print(f"  {'Weight zeroing':<25} {np.mean(zero_rhos):>8.3f} {'—':>10} "
              f"{compute_flip_rates(all_imp_zero, nl, nh)[0].mean():>8.3f} "
              f"{compute_flip_rates(all_imp_zero, nl, nh)[1].mean():>8.3f}")
        print(f"  {'Mean ablation':<25} {np.mean(mean_rhos):>8.3f} "
              f"{np.mean(ginv_mean):>10.3f} "
              f"{np.mean(within_m):>8.3f} {np.mean(between_m):>8.3f}")
        print(f"\n  Cross-method per-model ρ: {np.mean(cross):.3f} "
              f"[{np.min(cross):.3f}, {np.max(cross):.3f}]")
        print(f"  (High cross-method ρ = both methods measure the same structure)")

        config_results = {
            'config': cfg['name'],
            'weight_zeroing': {
                'full_rho': float(np.mean(zero_rhos)),
            },
            'mean_ablation': {
                'full_rho': float(np.mean(mean_rhos)),
                'ginv_rho': float(np.mean(ginv_mean)),
                'within_flip': float(np.mean(within_m)),
                'between_flip': float(np.mean(between_m)),
            },
            'cross_method': {
                'mean_rho': float(np.mean(cross)),
                'min_rho': float(np.min(cross)),
                'max_rho': float(np.max(cross)),
                'per_model': [float(x) for x in cross],
            },
        }
        all_results[config_key] = config_results

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY: Does the pattern hold under mean ablation?")
    print(f"{'=' * 60}")
    for k in ['A', 'B']:
        r = all_results[k]
        wz = r['weight_zeroing']['full_rho']
        ma = r['mean_ablation']['full_rho']
        gi = r['mean_ablation']['ginv_rho']
        wf = r['mean_ablation']['within_flip']
        bf = r['mean_ablation']['between_flip']
        cx = r['cross_method']['mean_rho']
        print(f"\n  Config {k}:")
        print(f"    Full ρ:  zero={wz:.3f}  mean={ma:.3f}  (same instability? {'YES' if abs(wz - ma) < 0.15 else 'NO'})")
        print(f"    G-inv ρ: {gi:.3f}  (lift present? {'YES' if gi > 0.8 else 'NO'})")
        print(f"    W-flip:  {wf:.3f}  (≈0.5? {'YES' if wf > 0.35 else 'NO'})")
        print(f"    B-flip:  {bf:.3f}  (≈0.0? {'YES' if bf < 0.15 else 'NO'})")
        print(f"    Cross:   {cx:.3f}  (methods agree? {'YES' if cx > 0.7 else 'NO'})")

    with open(OUT_DIR / 'results_mean_ablation_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to results_mean_ablation_comparison.json")


if __name__ == '__main__':
    main()
