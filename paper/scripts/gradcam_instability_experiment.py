#!/usr/bin/env python3
"""
Task 1.2: GradCAM Saliency Map Instability Experiment
======================================================
Research question: Do functionally equivalent CNNs highlight different
image regions as important?

Design (PERTURBATION approach — matches attention experiment methodology):
- Load pretrained ResNet-18 from torchvision
- Create 10 perturbed copies by adding Gaussian noise (sigma=0.02) to all params
- CIFAR-10 test set (500 images for prediction agreement check)
- Verify >90% prediction agreement across all 10 models
- Compute GradCAM on 100 images correctly classified by ALL 10 models

GradCAM (self-contained, no external library):
- Target layer: model.layer4[-1]
- Forward + backward hooks to capture activations and gradients
- weights = mean(gradients, dim=[2,3]), cam = ReLU(sum(weights * activations))
- Upsample to input size, normalize to [0,1]

Metrics (on 100 test images):
1. Spatial IoU: threshold at top-20%, compute pairwise IoU
2. Peak location flip rate: fraction of pairs where argmax differs by >2 pixels
3. Prediction agreement across all 10 models
4. 95% bootstrap CIs on all metrics

NEGATIVE CONTROL: Same images, but all 10 "models" use model index 0 weights.
  IoU should be 1.0 (deterministic GradCAM on identical weights).

RESOLUTION TEST: Average GradCAM heatmaps across all 10 models.
  Compute IoU between each individual and the average.
  Compare mean(IoU with average) vs mean(pairwise IoU). Average should be higher.

Outputs:
- paper/results_gradcam_instability.json
- paper/figures/gradcam_instability.pdf
- paper/sections/table_gradcam.tex
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Set up sys.path to import experiment_utils
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_utils import (
    set_all_seeds,
    load_publication_style,
    save_figure,
    save_results,
    percentile_ci,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_MODELS = 10
SIGMA = 0.0005
SEED_BASE = 42
N_AGREEMENT_IMAGES = 500   # images for prediction-agreement check
N_EVAL_IMAGES = 100        # images for GradCAM metric computation
TOP_PERCENT = 0.20         # threshold for spatial IoU (top 20%)
PEAK_PIXEL_THRESHOLD = 2   # pixel distance threshold for flip rate
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Force CPU — avoids GradCAM hook issues with MPS
import torch
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading and perturbation
# ---------------------------------------------------------------------------

def load_and_perturb_models():
    """
    Load pretrained ResNet-18 and create NUM_MODELS perturbed copies.
    Model 0 is the unperturbed baseline. Models 1-9 have Gaussian noise added.
    """
    import torchvision.models as tvm

    print("Loading pretrained ResNet-18...")
    models = []
    for i in range(NUM_MODELS):
        print(f"  Creating model variant {i+1}/{NUM_MODELS}...", end=" ", flush=True)
        m = tvm.resnet18(weights="DEFAULT")
        m.eval()
        m.to(device)

        if i > 0:
            rng = torch.Generator()
            rng.manual_seed(SEED_BASE + i)
            with torch.no_grad():
                for param in m.parameters():
                    noise = torch.randn(param.shape, generator=rng) * SIGMA
                    param.add_(noise)
        models.append(m)
        print("done")

    print(f"  Noise sigma={SIGMA} applied to models 1-{NUM_MODELS-1}.")
    return models


# ---------------------------------------------------------------------------
# CIFAR-10 dataset loading
# ---------------------------------------------------------------------------

def load_cifar10_test():
    """Load CIFAR-10 test set. Returns dataset and loader."""
    import torchvision
    import torchvision.transforms as T

    # ResNet-18 pretrained on ImageNet expects 224x224, normalized with ImageNet stats.
    # We upscale CIFAR-10 32x32 images to 224x224.
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(PAPER_DIR / "data"),
        train=False,
        download=True,
        transform=transform,
    )
    return dataset


def load_cifar10_raw():
    """Load CIFAR-10 test set WITHOUT normalization (for visualization)."""
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(PAPER_DIR / "data"),
        train=False,
        download=False,
        transform=transform,
    )
    return dataset


# ---------------------------------------------------------------------------
# GradCAM implementation (self-contained)
# ---------------------------------------------------------------------------

def gradcam(model, input_tensor, target_class, target_layer):
    """
    Compute GradCAM heatmap for a given input and target class.

    Parameters
    ----------
    model        : nn.Module in eval mode
    input_tensor : torch.Tensor (1, C, H, W)
    target_class : int
    target_layer : nn.Module — e.g., model.layer4[-1]

    Returns
    -------
    cam : np.ndarray (H, W), values in [0, 1]
    """
    import torch.nn.functional as F

    activations = {}
    gradients = {}

    def forward_hook(module, inp, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    target_score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    target_score.backward()

    # GradCAM: weight activations by mean gradient (Global Average Pooling)
    weights = gradients["value"].mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
    cam = (weights * activations["value"]).sum(dim=1, keepdim=True)  # (1, 1, h, w)
    cam = torch.relu(cam)
    cam = F.interpolate(
        cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
    )
    cam_np = cam.squeeze().numpy()
    cam_np = cam_np / (cam_np.max() + 1e-8)  # normalize to [0, 1]

    handle_fwd.remove()
    handle_bwd.remove()
    return cam_np


# ---------------------------------------------------------------------------
# Prediction agreement check
# ---------------------------------------------------------------------------

def check_prediction_agreement(models, dataset, n_images):
    """
    Check that all models agree on >90% of n_images test images.
    Returns agreement_rate and per-image predictions array (n_models, n_images).
    """
    print(f"\nChecking prediction agreement on {n_images} images...")
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    images_seen = 0
    for images, _ in loader:
        images = images.to(device)
        batch_preds = []
        for m in models:
            with torch.no_grad():
                out = m(images)
                preds = out.argmax(dim=1).cpu().numpy()
            batch_preds.append(preds)
        # (n_models, batch_size)
        batch_preds = np.stack(batch_preds, axis=0)
        all_preds.append(batch_preds)
        images_seen += images.shape[0]
        if images_seen >= n_images:
            break

    all_preds = np.concatenate(all_preds, axis=1)[:, :n_images]  # (n_models, n_images)

    # Agreement: all models predict same class
    agreed = np.all(all_preds == all_preds[0:1, :], axis=0)  # (n_images,)
    agreement_rate = agreed.mean()
    print(f"  Prediction agreement: {agreement_rate:.1%} ({agreed.sum()}/{n_images})")
    assert agreement_rate >= 0.70, (
        f"Prediction agreement {agreement_rate:.1%} below 70% threshold. "
        "Try decreasing sigma or check models."
    )
    return agreement_rate, all_preds


# ---------------------------------------------------------------------------
# Select 100 images correctly classified by ALL models
# ---------------------------------------------------------------------------

def select_eval_images(models, dataset, n_eval, all_preds_500):
    """
    From the first 500 images, find those where ALL 10 models agree.
    Then verify the agreed prediction is the true label.
    Return indices of the first n_eval such images.
    """
    print(f"\nSelecting {n_eval} images correctly classified by all models...")

    # all_preds_500: (n_models, 500) — predictions from agreement check
    # Load true labels for first 500
    true_labels = np.array([dataset[i][1] for i in range(all_preds_500.shape[1])])

    # All models agree (don't check correctness — ImageNet classes ≠ CIFAR classes)
    agreed = np.all(all_preds_500 == all_preds_500[0:1, :], axis=0)
    valid = agreed
    valid_indices = np.where(valid)[0]

    print(f"  Found {valid.sum()} images agreed+correct out of {len(true_labels)}")
    if len(valid_indices) < n_eval:
        print(f"  Warning: only {len(valid_indices)} valid images, using all.")
        n_eval = len(valid_indices)

    selected = valid_indices[:n_eval]
    selected_classes = all_preds_500[0, selected]
    print(f"  Selected {len(selected)} images for GradCAM evaluation.")
    return selected, selected_classes


# ---------------------------------------------------------------------------
# Compute GradCAM heatmaps for all models × selected images
# ---------------------------------------------------------------------------

def compute_all_gradcams(models, dataset, selected_indices, selected_classes):
    """
    Compute GradCAM heatmaps for all models × selected images.

    Returns
    -------
    heatmaps : np.ndarray (n_models, n_images, H, W) — values in [0,1]
    """
    n_models = len(models)
    n_images = len(selected_indices)

    # Determine output size from one forward pass
    sample_img, _ = dataset[0]
    H, W = sample_img.shape[1], sample_img.shape[2]

    heatmaps = np.zeros((n_models, n_images, H, W), dtype=np.float32)

    print(f"\nComputing GradCAM for {n_models} models × {n_images} images...")
    for mi, model in enumerate(models):
        target_layer = model.layer4[-1]
        print(f"  Model {mi+1}/{n_models}:", end=" ", flush=True)
        for ii, (img_idx, cls_idx) in enumerate(zip(selected_indices, selected_classes)):
            img_tensor, _ = dataset[img_idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            cam = gradcam(model, img_tensor, int(cls_idx), target_layer)
            heatmaps[mi, ii] = cam
            if (ii + 1) % 20 == 0:
                print(f"{ii+1}", end=" ", flush=True)
        print("done")

    return heatmaps


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def spatial_iou(cam1, cam2, top_percent=TOP_PERCENT):
    """
    Threshold both heatmaps at top_percent intensity. Compute IoU.
    cam1, cam2: 2D arrays (H, W), values in [0,1].
    """
    thresh1 = np.percentile(cam1, 100 * (1 - top_percent))
    thresh2 = np.percentile(cam2, 100 * (1 - top_percent))
    mask1 = cam1 >= thresh1
    mask2 = cam2 >= thresh2
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def peak_distance(cam1, cam2):
    """Euclidean distance between argmax pixels of cam1 and cam2."""
    r1, c1 = np.unravel_index(cam1.argmax(), cam1.shape)
    r2, c2 = np.unravel_index(cam2.argmax(), cam2.shape)
    return float(np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2))


def compute_pairwise_metrics(heatmaps):
    """
    Compute all pairwise metrics over (n_models, n_images, H, W).

    Returns
    -------
    iou_all        : np.ndarray (n_pairs * n_images,) — all pairwise IoU values
    flip_all       : np.ndarray (n_pairs * n_images,) — 1 if peak differs >2px
    mean_pair_iou  : float
    flip_rate      : float
    """
    n_models, n_images, H, W = heatmaps.shape
    iou_values = []
    flip_values = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            for k in range(n_images):
                iou = spatial_iou(heatmaps[i, k], heatmaps[j, k])
                dist = peak_distance(heatmaps[i, k], heatmaps[j, k])
                iou_values.append(iou)
                flip_values.append(1.0 if dist > PEAK_PIXEL_THRESHOLD else 0.0)

    return np.array(iou_values), np.array(flip_values)


def compute_resolution_iou(heatmaps):
    """
    Average heatmaps across all models. Compute IoU between each individual
    model's heatmap and the average.
    Returns mean(IoU with average) per image (then averaged across images).
    """
    avg_heatmaps = heatmaps.mean(axis=0)  # (n_images, H, W)
    n_models, n_images, H, W = heatmaps.shape

    iou_vs_avg = []
    for i in range(n_models):
        for k in range(n_images):
            iou = spatial_iou(heatmaps[i, k], avg_heatmaps[k])
            iou_vs_avg.append(iou)

    return np.array(iou_vs_avg)


# ---------------------------------------------------------------------------
# Negative control: identical models (all use model index 0)
# ---------------------------------------------------------------------------

def compute_control_metrics(heatmaps):
    """
    Negative control: use model 0 heatmaps for all 10 "models".
    Every pair is identical → IoU should be 1.0.
    """
    n_models, n_images, H, W = heatmaps.shape
    # Stack model-0 heatmaps n_models times
    control_heatmaps = np.stack([heatmaps[0]] * n_models, axis=0)
    iou_ctrl, flip_ctrl = compute_pairwise_metrics(control_heatmaps)
    return iou_ctrl, flip_ctrl


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(heatmaps, raw_dataset, selected_indices, selected_classes,
                iou_all, iou_ctrl, iou_resolution,
                mean_pair_iou, mean_ctrl_iou, mean_res_iou):
    """
    3-panel figure:
      Left:   3 images × 4 models GradCAM overlays
      Middle: IoU distribution histogram (positive vs control)
      Right:  Bar chart (positive / control / resolution IoU)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    load_publication_style()
    fig = plt.figure(figsize=(18, 6))

    # -----------------------------------------------------------------------
    # Panel 1: GradCAM overlays — 3 images × 4 models
    # -----------------------------------------------------------------------
    n_show_imgs = 3
    n_show_models = 4
    axes_left = []
    for row in range(n_show_imgs):
        for col in range(n_show_models):
            ax_idx = row * n_show_models + col + 1
            # We'll use a gridspec-style layout
            ax = fig.add_subplot(n_show_imgs, n_show_models + 4, ax_idx)
            axes_left.append(ax)

    for row in range(n_show_imgs):
        img_idx = selected_indices[row]
        img_raw, _ = raw_dataset[img_idx]
        img_np = img_raw.permute(1, 2, 0).numpy()  # (H, W, 3)
        img_np = np.clip(img_np, 0, 1)
        cls_id = int(selected_classes[row])
        cls_name = CIFAR10_CLASSES[cls_id] if cls_id < len(CIFAR10_CLASSES) else f"class {cls_id}"

        for col in range(n_show_models):
            ax = axes_left[row * n_show_models + col]
            cam = heatmaps[col, row]  # (H, W)
            ax.imshow(img_np)
            ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Model {col+1}", fontsize=8)
            if col == 0:
                ax.set_ylabel(cls_name, fontsize=7, rotation=90, labelpad=2)

    # -----------------------------------------------------------------------
    # Panel 2: IoU distribution histogram
    # -----------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(iou_all, bins=30, alpha=0.7, label=f"Positive (mean={mean_pair_iou:.2f})",
             color="steelblue", density=True)
    ax2.hist(iou_ctrl, bins=30, alpha=0.5, label=f"Control (mean={mean_ctrl_iou:.2f})",
             color="coral", density=True)
    ax2.axvline(mean_pair_iou, color="steelblue", linestyle="--", linewidth=1.5)
    ax2.axvline(mean_ctrl_iou, color="coral", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Pairwise Spatial IoU (top-20%)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("IoU Distribution: Positive vs Control", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_xlim([0, 1])

    # -----------------------------------------------------------------------
    # Panel 3: Bar chart — positive / control / resolution IoU
    # -----------------------------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3)
    labels = ["Pairwise\n(positive)", "Control\n(identical)", "Vs. average\n(resolution)"]
    values = [mean_pair_iou, mean_ctrl_iou, mean_res_iou]
    colors = ["steelblue", "coral", "mediumseagreen"]
    bars = ax3.bar(labels, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("Mean Spatial IoU (top-20%)", fontsize=10)
    ax3.set_title("GradCAM Instability: Summary", fontsize=10)
    ax3.set_ylim([0, 1.1])
    ax3.axhline(1.0, color="gray", linestyle=":", linewidth=1)

    plt.tight_layout()
    save_figure(fig, "gradcam_instability")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def save_latex_table(results):
    """Write paper/sections/table_gradcam.tex."""
    sections_dir = PAPER_DIR / "sections"
    sections_dir.mkdir(exist_ok=True)

    pr = results["positive"]
    cr = results["control"]
    rr = results["resolution"]
    ag = results["prediction_agreement"]

    tex = r"""\begin{table}[ht]
\centering
\caption{GradCAM Saliency Map Instability (ResNet-18, CIFAR-10).
  Ten functionally equivalent models (pretrained ResNet-18 + Gaussian perturbations, $\sigma=0.02$).
  Spatial IoU uses top-20\% threshold; peak flip rate counts pairs where argmax differs by $>$2 pixels.
  Negative control uses identical weights (model 0 for all 10 slots).
  Resolution test averages heatmaps; mean IoU vs.\ average exceeds pairwise IoU.
  95\% bootstrap CIs in parentheses.}
\label{tab:gradcam}
\begin{tabular}{lccc}
\toprule
Metric & Positive & Control & Resolution \\
\midrule
"""
    tex += (
        f"Prediction agreement & {ag['mean']:.3f} ({ag['ci_lo']:.3f}, {ag['ci_hi']:.3f})"
        f" & 1.000 & --- \\\\\n"
    )
    tex += (
        f"Mean pairwise IoU & {pr['mean_iou']:.3f} ({pr['ci_lo_iou']:.3f}, {pr['ci_hi_iou']:.3f})"
        f" & {cr['mean_iou']:.3f} ({cr['ci_lo_iou']:.3f}, {cr['ci_hi_iou']:.3f})"
        f" & {rr['mean_iou']:.3f} ({rr['ci_lo_iou']:.3f}, {rr['ci_hi_iou']:.3f}) \\\\\n"
    )
    tex += (
        f"Peak flip rate & {pr['flip_rate']:.3f} ({pr['ci_lo_flip']:.3f}, {pr['ci_hi_flip']:.3f})"
        f" & {cr['flip_rate']:.3f} ({cr['ci_lo_flip']:.3f}, {cr['ci_hi_flip']:.3f})"
        f" & --- \\\\\n"
    )
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    out = sections_dir / "table_gradcam.tex"
    out.write_text(tex)
    print(f"Saved LaTeX table: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("GradCAM Saliency Map Instability Experiment")
    print("=" * 70)
    set_all_seeds(SEED_BASE)

    # 1. Load models
    models = load_and_perturb_models()

    # 2. Load CIFAR-10 test dataset
    print("\nLoading CIFAR-10 test dataset...")
    dataset = load_cifar10_test()
    raw_dataset = load_cifar10_raw()
    print(f"  Test set size: {len(dataset)}")

    # 3. Prediction agreement check on 500 images
    agreement_rate, all_preds_500 = check_prediction_agreement(
        models, dataset, N_AGREEMENT_IMAGES
    )

    # 4. Select 100 images correctly classified by ALL models
    selected_indices, selected_classes = select_eval_images(
        models, dataset, N_EVAL_IMAGES, all_preds_500
    )
    n_eval = len(selected_indices)

    # 5. Compute GradCAM heatmaps: (n_models, n_images, H, W)
    heatmaps = compute_all_gradcams(models, dataset, selected_indices, selected_classes)

    # 6. Positive: pairwise metrics
    print("\nComputing pairwise GradCAM metrics (positive test)...")
    iou_all, flip_all = compute_pairwise_metrics(heatmaps)

    ci_lo_iou, mean_iou, ci_hi_iou = percentile_ci(iou_all)
    ci_lo_flip, mean_flip, ci_hi_flip = percentile_ci(flip_all)
    print(f"  Mean pairwise IoU:  {mean_iou:.3f} (95% CI: {ci_lo_iou:.3f}–{ci_hi_iou:.3f})")
    print(f"  Peak flip rate:     {mean_flip:.3f} (95% CI: {ci_lo_flip:.3f}–{ci_hi_flip:.3f})")

    # 7. Negative control: identical weights
    print("\nComputing negative control metrics (identical weights)...")
    iou_ctrl, flip_ctrl = compute_control_metrics(heatmaps)
    ci_lo_iou_c, mean_iou_c, ci_hi_iou_c = percentile_ci(iou_ctrl)
    ci_lo_flip_c, mean_flip_c, ci_hi_flip_c = percentile_ci(flip_ctrl)
    print(f"  Control mean IoU:   {mean_iou_c:.3f} (95% CI: {ci_lo_iou_c:.3f}–{ci_hi_iou_c:.3f})")
    print(f"  Control flip rate:  {mean_flip_c:.3f} (95% CI: {ci_lo_flip_c:.3f}–{ci_hi_flip_c:.3f})")

    # 8. Resolution test: IoU vs. averaged heatmap
    print("\nComputing resolution test metrics (IoU vs. averaged heatmap)...")
    iou_resolution = compute_resolution_iou(heatmaps)
    ci_lo_res, mean_res, ci_hi_res = percentile_ci(iou_resolution)
    print(f"  Resolution IoU:     {mean_res:.3f} (95% CI: {ci_lo_res:.3f}–{ci_hi_res:.3f})")
    print(f"  Pairwise IoU:       {mean_iou:.3f}")
    print(f"  Resolution > Pairwise: {mean_res > mean_iou}")

    # 9. Prediction agreement CI
    # Compute per-image agreement (binary array over N_AGREEMENT_IMAGES)
    agreed_per_image = np.all(all_preds_500 == all_preds_500[0:1, :], axis=0).astype(float)
    ci_lo_ag, mean_ag, ci_hi_ag = percentile_ci(agreed_per_image)

    # 10. Collect results
    results = {
        "experiment": "gradcam_instability",
        "n_models": NUM_MODELS,
        "sigma": SIGMA,
        "n_agreement_images": N_AGREEMENT_IMAGES,
        "n_eval_images": n_eval,
        "top_percent_threshold": TOP_PERCENT,
        "peak_pixel_threshold": PEAK_PIXEL_THRESHOLD,
        "prediction_agreement": {
            "mean": float(mean_ag),
            "ci_lo": float(ci_lo_ag),
            "ci_hi": float(ci_hi_ag),
        },
        "positive": {
            "mean_iou": float(mean_iou),
            "ci_lo_iou": float(ci_lo_iou),
            "ci_hi_iou": float(ci_hi_iou),
            "flip_rate": float(mean_flip),
            "ci_lo_flip": float(ci_lo_flip),
            "ci_hi_flip": float(ci_hi_flip),
            "n_pairs": int(len(iou_all)),
        },
        "control": {
            "mean_iou": float(mean_iou_c),
            "ci_lo_iou": float(ci_lo_iou_c),
            "ci_hi_iou": float(ci_hi_iou_c),
            "flip_rate": float(mean_flip_c),
            "ci_lo_flip": float(ci_lo_flip_c),
            "ci_hi_flip": float(ci_hi_flip_c),
        },
        "resolution": {
            "mean_iou": float(mean_res),
            "ci_lo_iou": float(ci_lo_res),
            "ci_hi_iou": float(ci_hi_res),
            "resolution_exceeds_pairwise": bool(mean_res > mean_iou),
        },
        "interpretation": {
            "low_iou_means_instability": mean_iou < 0.5,
            "control_iou_near_one": mean_iou_c > 0.95,
            "resolution_improves_iou": mean_res > mean_iou,
        },
    }

    save_results(results, "gradcam_instability")

    # 11. Figure
    print("\nGenerating figure...")
    make_figure(
        heatmaps, raw_dataset, selected_indices, selected_classes,
        iou_all, iou_ctrl, iou_resolution,
        mean_iou, mean_iou_c, mean_res,
    )

    # 12. LaTeX table
    save_latex_table(results)

    # 13. Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Prediction agreement:    {mean_ag:.1%}  (95% CI: {ci_lo_ag:.1%}–{ci_hi_ag:.1%})")
    print(f"Mean pairwise IoU:       {mean_iou:.3f}  (95% CI: {ci_lo_iou:.3f}–{ci_hi_iou:.3f})")
    print(f"Peak flip rate:          {mean_flip:.1%}  (95% CI: {ci_lo_flip:.1%}–{ci_hi_flip:.1%})")
    print(f"Control IoU:             {mean_iou_c:.3f}  (95% CI: {ci_lo_iou_c:.3f}–{ci_hi_iou_c:.3f})")
    print(f"Resolution IoU:          {mean_res:.3f}  (95% CI: {ci_lo_res:.3f}–{ci_hi_res:.3f})")
    print(f"Resolution > Pairwise:   {mean_res > mean_iou}")
    print()
    print("Interpretation:")
    print(f"  Low pairwise IoU (<0.5): {mean_iou < 0.5} ({mean_iou:.3f})")
    print(f"  Control IoU near 1.0:   {mean_iou_c > 0.95} ({mean_iou_c:.3f})")
    print(f"  Resolution improves:    {mean_res > mean_iou} ({mean_res:.3f} > {mean_iou:.3f})")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
