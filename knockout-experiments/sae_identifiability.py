#!/usr/bin/env python
"""
SAE feature identifiability (roadmap #7 extension; tests a monograph OPEN question).

The monograph (Sec. on SAEs) states: "If independently trained SAEs on the same model
produce stable features, SAE-based attribution may escape the Rashomon property -- the
symmetry group would be trivial (G={e}) and eta=0. Whether this holds empirically is
untested." This experiment tests it directly on freshly grokked transformers (GPU/MPS).

Method: train transformers on (a+b) mod P; collect MLP activations; train K sparse
autoencoders (different seeds) on the SAME activations; match dictionary features across
seeds by decoder-direction cosine (Hungarian) and measure the matched-cosine distribution
and the STABLE-FEATURE FRACTION (features reproduced above a threshold). If that fraction
is < 1, SAE features are Rashomon-unstable across training randomness -> SAEs do NOT
escape the impossibility. Controls: reconstruction R^2 (are the SAEs any good?) and a
random-direction null (matched cosine of random unit vectors).
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from mod_add_interp_identifiability import Transformer, make_data, DEV, P, D_MLP

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_sae_identifiability.json")
N_MODELS = 2
N_SAE_PER_MODEL = 6
D_SAE = 1024
L1 = 2e-3
SAE_STEPS = 4000
TF_MAX_STEPS, TF_EVAL, TF_TARGET = 20000, 250, 0.99


def train_transformer(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X, Y = make_data()
    n = X.shape[0]; perm = np.random.permutation(n); ntr = n // 2
    Xtr, Ytr = X[perm[:ntr]].to(DEV), Y[perm[:ntr]].to(DEV)
    Xte, Yte = X[perm[ntr:]].to(DEV), Y[perm[ntr:]].to(DEV)
    m = Transformer(P).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    acc = 0.0
    for step in range(TF_MAX_STEPS + 1):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(Xtr), Ytr).backward(); opt.step()
        if step % TF_EVAL == 0:
            m.eval()
            with torch.no_grad():
                acc = (m(Xte).argmax(-1) == Yte).float().mean().item()
            if acc >= TF_TARGET:
                break
    with torch.no_grad():
        _, acts = m(X.to(DEV), return_acts=True)
    return acts.float().to(DEV), acc


class SAE(nn.Module):
    def __init__(self, d_act, d_sae):
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(d_act))
        self.enc = nn.Linear(d_act, d_sae)
        self.dec = nn.Linear(d_sae, d_act, bias=False)
        with torch.no_grad():
            self.dec.weight.div_(self.dec.weight.norm(dim=0, keepdim=True) + 1e-8)

    def forward(self, x):
        z = F.relu(self.enc(x - self.b_pre))
        return self.dec(z) + self.b_pre, z


def train_sae(acts, seed):
    torch.manual_seed(seed)
    sae = SAE(acts.shape[1], D_SAE).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for step in range(SAE_STEPS):
        opt.zero_grad()
        xhat, z = sae(acts)
        loss = F.mse_loss(xhat, acts) + L1 * z.abs().sum(-1).mean()
        loss.backward(); opt.step()
        with torch.no_grad():                       # unit-norm decoder columns (standard SAE)
            sae.dec.weight.div_(sae.dec.weight.norm(dim=0, keepdim=True) + 1e-8)
    with torch.no_grad():
        xhat, z = sae(acts)
        var = ((acts - acts.mean(0)) ** 2).sum().item()
        r2 = 1.0 - ((acts - xhat) ** 2).sum().item() / (var + 1e-9)
        l0 = (z > 1e-6).float().sum(-1).mean().item()
        D = sae.dec.weight.detach().cpu().numpy().T          # (d_sae, d_act) unit rows
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    return {"D": D, "r2": r2, "l0": l0}


def match_stats(Da, Db):
    C = np.abs(Da @ Db.T)                # |cosine| (d_sae x d_sae)
    r, c = linear_sum_assignment(-C)
    m = C[r, c]
    return m


def analyze(per_model):
    thr = [0.9, 0.7, 0.5]
    all_matched, r2s, l0s = [], [], []
    for saes in per_model:
        for s in saes:
            r2s.append(s["r2"]); l0s.append(s["l0"])
        for i in range(len(saes)):
            for j in range(i + 1, len(saes)):
                all_matched.append(match_stats(saes[i]["D"], saes[j]["D"]))
    matched = np.concatenate(all_matched)
    # random-direction null
    rng = np.random.default_rng(0)
    d_act = per_model[0][0]["D"].shape[1]
    A = rng.standard_normal((D_SAE, d_act)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    B = rng.standard_normal((D_SAE, d_act)); B /= np.linalg.norm(B, axis=1, keepdims=True)
    null = match_stats(A, B)
    return {
        "n_models": len(per_model), "n_sae_per_model": len(per_model[0]),
        "d_act": int(d_act), "d_sae": D_SAE,
        "sae_reconstruction_r2": {"mean": round(float(np.mean(r2s)), 4), "min": round(float(np.min(r2s)), 4)},
        "sae_l0_sparsity_mean": round(float(np.mean(l0s)), 2),
        "matched_cosine_across_seeds": {"mean": round(float(matched.mean()), 4),
                                        "median": round(float(np.median(matched)), 4),
                                        "min": round(float(matched.min()), 4)},
        "STABLE_feature_fraction": {f"cos>{t}": round(float((matched > t).mean()), 4) for t in thr},
        "random_direction_null_matched_cosine_mean": round(float(null.mean()), 4),
        "interpretation": ("Fraction of SAE dictionary features reproduced across independent SAE "
                           "training seeds on the SAME model. If < 1, SAE features are Rashomon-"
                           "unstable and do NOT escape the impossibility; the identifiable content is "
                           "the stable-feature subset, and the rest is training-seed noise -- exactly "
                           "the framework's prediction (G is nontrivial, eta > 0)."),
    }


def main():
    print(f"device={DEV}, {N_MODELS} transformers x {N_SAE_PER_MODEL} SAEs (d_sae={D_SAE})")
    t0 = time.time()
    per_model = []
    for mi in range(N_MODELS):
        acts, acc = train_transformer(seed=2000 + mi)
        saes = [train_sae(acts, seed=10 * mi + s) for s in range(N_SAE_PER_MODEL)]
        per_model.append(saes)
        print(f"  model {mi}: test_acc={acc:.3f}, SAE R2 mean="
              f"{np.mean([s['r2'] for s in saes]):.3f}, L0={np.mean([s['l0'] for s in saes]):.1f} "
              f"({time.time()-t0:.0f}s)")
    summary = analyze(per_model)
    json.dump({"summary": summary}, open(OUT, "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
