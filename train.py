"""
train.py

Main training script. Trains and compares:
  1. BaselineSNN  -- pure backpropagation, no plasticity, no replay.
  2. HybridSNN    -- backpropagation (slow pathway) + local delta-rule
                     plasticity with episodic replay (fast pathway).

Both models are trained sequentially on the 5 Split-MNIST tasks
(class-incremental setting) and evaluated for catastrophic forgetting.

Usage:
    pip install torch torchvision
    python train.py
"""

import torch
import torch.nn as nn
import numpy as np

from data_utils import get_split_mnist_tasks, rate_encode, NUM_STEPS, collect_exemplars
from baseline_model import BaselineSNN
from hybrid_model import HybridSNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_FEATURES = 784
HIDDEN_FEATURES = 128
OUT_FEATURES = 10
EPOCHS_PER_TASK = 2
LR = 1e-3


def evaluate(model, test_loaders, task_ids, is_hybrid):
    """Computes accuracy on each of the given (already-learned) tasks."""
    model.eval()
    accs = []
    with torch.no_grad():
        for tid in task_ids:
            correct, total = 0, 0
            for images, labels in test_loaders[tid]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                spikes_in = rate_encode(images).to(DEVICE)

                if is_hybrid:
                    model.reset_fast_pathway(images.shape[0], DEVICE)
                    out = model(spikes_in, update_plasticity=False)
                else:
                    out = model(spikes_in)

                pred = out.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.shape[0]
            accs.append(correct / total)
    model.train()
    return accs


def train_continual(model, train_loaders, test_loaders, is_hybrid, tag, use_replay=False):
    """
    Trains a model sequentially across all tasks in train_loaders, and
    evaluates it on all tasks seen so far after each task is trained.

    Returns:
        accuracy_matrix: accuracy_matrix[i] = accuracy on tasks 0..i,
                          measured right after task i is trained.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    accuracy_matrix = []
    exemplar_buffer = []  # episodic replay memory: list of (images, labels)

    for task_id, loader in enumerate(train_loaders):
        print(f"\n[{tag}] Training on Task {task_id + 1}/5 ...")
        for epoch in range(EPOCHS_PER_TASK):
            for batch_idx, (images, labels) in enumerate(loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                spikes_in = rate_encode(images).to(DEVICE)

                if is_hybrid:
                    model.reset_fast_pathway(images.shape[0], DEVICE)
                    out = model(spikes_in, update_plasticity=True, labels=labels)
                else:
                    out = model(spikes_in)

                loss = loss_fn(out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Replay: periodically re-expose the fast pathway to
                # exemplars from earlier tasks. The slow pathway is
                # unaffected.
                if is_hybrid and use_replay and exemplar_buffer and batch_idx % 5 == 0:
                    for buf_imgs, buf_labels in exemplar_buffer:
                        buf_imgs, buf_labels = buf_imgs.to(DEVICE), buf_labels.to(DEVICE)
                        buf_spikes = rate_encode(buf_imgs).to(DEVICE)
                        model.reset_fast_pathway(buf_imgs.shape[0], DEVICE)
                        model.fast_pathway_replay(buf_spikes, buf_labels,
                                                   num_classes=OUT_FEATURES)

        seen_tasks = list(range(task_id + 1))
        accs = evaluate(model, test_loaders, seen_tasks, is_hybrid)
        accuracy_matrix.append(accs)
        print(f"[{tag}] After Task {task_id + 1}, accuracy on seen tasks: "
              f"{[round(a, 3) for a in accs]}")

        if is_hybrid and getattr(model, "use_gate", True):
            gate_value = torch.sigmoid(model.gate).item()
            print(f"[{tag}] Current gate value (fast pathway weight): "
                  f"{gate_value:.3f}  (0 = all slow, 1 = all fast)")

        # Store a small exemplar set from this task for future replay.
        if is_hybrid and use_replay:
            exemplar_buffer.append(collect_exemplars(loader, num_samples=20))

    return accuracy_matrix


def compute_forgetting(accuracy_matrix):
    """
    Average forgetting: for each task (except the last), the drop in
    accuracy between when it was first learned and after all tasks have
    been trained.
    """
    n_tasks = len(accuracy_matrix)
    final_accs = accuracy_matrix[-1]
    drops = []
    for i in range(n_tasks - 1):
        acc_when_learned = accuracy_matrix[i][i]
        acc_at_end = final_accs[i]
        drops.append(acc_when_learned - acc_at_end)
    return np.mean(drops) if drops else 0.0


def compute_bwt(accuracy_matrix):
    """
    Backward Transfer (BWT), a standard continual learning metric
    (Lopez-Paz & Ranzato, 2017): BWT = average(Acc_final,i - Acc_i,i).
    Negative BWT indicates forgetting; values closer to 0 are better.
    Equivalent to -1 * average forgetting as defined above.
    """
    return -compute_forgetting(accuracy_matrix)


def diagnose_pathways(model, test_loaders, task_ids):
    """
    Diagnostic: prints the slow and fast pathways' accuracy separately,
    to identify which pathway is responsible for retaining or forgetting
    a given task.
    """
    model.eval()
    print("\n[DIAGNOSTIC] Isolated pathway accuracy per task:")
    with torch.no_grad():
        for tid in task_ids:
            slow_correct, fast_correct, total = 0, 0, 0
            for images, labels in test_loaders[tid]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                spikes_in = rate_encode(images).to(DEVICE)
                slow_out, fast_out = model.forward_debug(spikes_in)

                slow_correct += (slow_out.argmax(dim=1) == labels).sum().item()
                fast_correct += (fast_out.argmax(dim=1) == labels).sum().item()
                total += labels.shape[0]

            print(f"  Task {tid + 1}: slow-only acc = {slow_correct/total:.3f}, "
                  f"fast-only acc = {fast_correct/total:.3f}")
    model.train()


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train_loaders, test_loaders, task_classes = get_split_mnist_tasks()

    print("\n===== BASELINE MODEL (pure backpropagation, no plasticity) =====")
    baseline = BaselineSNN(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(DEVICE)
    baseline_matrix = train_continual(baseline, train_loaders, test_loaders,
                                       is_hybrid=False, tag="BASELINE")
    baseline_forgetting = compute_forgetting(baseline_matrix)

    print("\n===== HYBRID MODEL (backpropagation + local plasticity + replay) =====")
    hybrid = HybridSNN(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(DEVICE)
    hybrid_matrix = train_continual(hybrid, train_loaders, test_loaders,
                                     is_hybrid=True, tag="HYBRID", use_replay=True)
    hybrid_forgetting = compute_forgetting(hybrid_matrix)

    print("\n===== FINAL RESULT =====")
    print(f"Baseline average forgetting: {baseline_forgetting:.4f}  |  BWT: {compute_bwt(baseline_matrix):.4f}")
    print(f"Hybrid   average forgetting: {hybrid_forgetting:.4f}  |  BWT: {compute_bwt(hybrid_matrix):.4f}")
    print("(lower forgetting / BWT closer to 0 is better)")

    diagnose_pathways(hybrid, test_loaders, list(range(5)))
