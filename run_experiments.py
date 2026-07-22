"""
run_experiments.py

Runs the full multi-seed ablation study:
  - 4 model variants: baseline, hybrid (full), hybrid (no replay),
    hybrid (no gate).
  - 5 random seeds per variant, for statistical reliability.
  - Reports mean +/- standard deviation of forgetting and Backward
    Transfer (BWT) for each variant.

Note: this runs 4 x 5 = 20 full training runs and can take 30-60+
minutes on a free-tier Colab GPU.

Usage:
    python run_experiments.py
"""

import torch
import numpy as np

from data_utils import get_split_mnist_tasks
from baseline_model import BaselineSNN
from hybrid_model import HybridSNN
from train import (train_continual, compute_forgetting, compute_bwt,
                    DEVICE, IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES)

SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["baseline", "hybrid_full", "hybrid_no_replay", "hybrid_no_gate"]


def build_model(variant):
    if variant == "baseline":
        return BaselineSNN(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(DEVICE), False, False
    elif variant == "hybrid_full":
        return HybridSNN(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(DEVICE), True, True
    elif variant == "hybrid_no_replay":
        return HybridSNN(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(DEVICE), True, False
    elif variant == "hybrid_no_gate":
        return HybridSNN(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES,
                          use_gate=False).to(DEVICE), True, True
    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_one(variant, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loaders, test_loaders, _ = get_split_mnist_tasks()
    model, is_hybrid, use_replay = build_model(variant)
    tag = f"{variant}-seed{seed}"
    matrix = train_continual(model, train_loaders, test_loaders,
                              is_hybrid=is_hybrid, tag=tag, use_replay=use_replay)

    return compute_forgetting(matrix), compute_bwt(matrix)


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    all_results = {v: {"forgetting": [], "bwt": []} for v in VARIANTS}

    for variant in VARIANTS:
        for seed in SEEDS:
            print(f"\n{'='*60}\nRunning variant='{variant}' seed={seed}\n{'='*60}")
            forgetting, bwt = run_one(variant, seed)
            all_results[variant]["forgetting"].append(forgetting)
            all_results[variant]["bwt"].append(bwt)
            print(f"  -> forgetting={forgetting:.4f}, BWT={bwt:.4f}")

    print("\n\n" + "=" * 70)
    print(f"FINAL SUMMARY (mean +/- std over {len(SEEDS)} seeds)")
    print("=" * 70)
    print(f"{'Variant':22s} | {'Forgetting':20s} | {'BWT':20s}")
    print("-" * 70)
    for variant in VARIANTS:
        f_arr = np.array(all_results[variant]["forgetting"])
        b_arr = np.array(all_results[variant]["bwt"])
        print(f"{variant:22s} | {f_arr.mean():.4f} +/- {f_arr.std():.4f}     "
              f"| {b_arr.mean():.4f} +/- {b_arr.std():.4f}")
