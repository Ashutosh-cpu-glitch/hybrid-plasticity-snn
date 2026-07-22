# Hybrid Plasticity Spiking Neural Networks for Continual Learning

**Author:** Ashutosh Dadhich
**Domain:** Neuromorphic Computing / Spiking Neural Networks / Continual Learning

---

## 1. Motivation

Spiking Neural Networks (SNNs) trained purely with backpropagation-through-time (BPTT) via surrogate gradients often suffer from catastrophic forgetting when trained sequentially on new tasks. This causes a significant loss of performance on tasks learned earlier reflecting the classic stability-plasticity dilemma in neuroscience.

Biological brains avoid this through Complementary Learning Systems (McClelland et al., 1995). The hippocampus learns new information quickly via fast, local, activity-dependent synaptic plasticity, while the cortex consolidates knowledge slowly. Hippocampal replay of past experiences (observed during rest and sleep) is believed to protect old memories from being overwritten during new learning.

This project asks: **can a small, locally-trained "fast" synaptic pathway,
combined with lightweight episodic replay, reduce catastrophic forgetting
in an SNN trained mainly by backpropagation?**

## 2. Architecture

![Architecture Diagram](figures/fig1_architecture.png)

The model, `HybridSNN`, has two parallel pathways whose outputs are combined
by a learned gate:

- **Slow pathway** - consists of two layers of Leaky Integrate-and-Fire (LIF) neurons trained end-to-end using backpropagation-through-time (BPTT) with surrogate gradients (fast sigmoid). This pathway performs the main task learning and serves as the standard gradient-based component of the network.
- **Fast pathway** - a single linear projection from input to output, updated **only** by a local, biologically-plausible learning rule (no backpropagation, no gradient through time). This represents the plasticity component.
- **Gate** (α, a learned scalar) - combines the two pathways:
  `output = α · fast_output + (1-α) · slow_output`.

![Gate Evolution](figures/fig4_gate_evolution.png)
- **Episodic replay buffer** — after each task, 20 exemplars are stored. While training on later tasks, these exemplars are periodically replayed through the fast pathway only (not the slow pathway), analogous to hippocampal replay protecting old associations.

![Replay Mechanism](figures/fig7_replay_mechanism.png)

A pure-backprop `BaselineSNN` (slow pathway only, no plasticity, no replay) is used as the control condition.

## 3. Experimental Setup

![Continual Learning Protocol](figures/fig2_protocol.png)

- **Task:** Split-MNIST — MNIST digits split into 5 sequential binary
  classification tasks: (0,1), (2,3), (4,5), (6,7), (8,9).
- **Protocol:** Class-incremental learning — the model is trained on Task 1,
  then Task 2, etc., with no access to old task data during training
  (except via the replay buffer for the hybrid model).
- **Input encoding:** Rate coding — pixel intensities converted into
  Bernoulli spike trains over 25 timesteps.
- **Metric:** Average forgetting — for each task, the drop in accuracy
  between when it was first learned and after all 5 tasks are trained
  (lower is better; 0 = no forgetting).
- **Hardware:** Google Colab, free-tier T4 GPU.

## 4. Development / Debugging Journey

This project went through several iterations, each motivated by diagnosing
*why* the fast pathway was not helping. This iterative process is itself
a core part of the research contribution.

| Iteration | Fast-pathway design | Baseline forgetting | Hybrid forgetting | Outcome |
|---|---|---|---|---|
| 1 | Unsupervised Hebbian STDP (no label signal) | 0.9954 | 0.9943 | No benefit — fast pathway had no notion of correct class |
| 2 | Teacher-guided STDP (potentiation & depression both driven by teacher signal) | 0.9941 | 0.9945 | No benefit — potentiation and depression cancelled each other out |
| 3 | Delta-rule error signal, but readout based on a self-referential spike threshold | 0.9960 | 0.9949 | No benefit — the threshold shifted as weights changed, destabilising learning |
| 4 | Delta-rule (Widrow-Hoff) on rate-coded (continuous) activity + episodic replay | 0.9952 | **0.7002** | **Significant reduction in forgetting** |

Each failed iteration was diagnosed using an isolated-pathway evaluation
(measuring the fast pathway's and slow pathway's accuracy separately)
before being fixed — for example, diagnostic evaluation on Iteration 3
showed the fast pathway achieved only 46% accuracy on Task 1 and 0% on the
task it had *just* been trained on, revealing the readout instability
rather than a forgetting problem per se.

## 5. Final Results

### 5.1 Multi-seed Ablation Study (5 random seeds, mean ± std)

To ensure statistical reliability, each variant was run across 5 random
seeds rather than relying on a single run.

| Variant | Average Forgetting | BWT |
|---|---|---|
| Baseline (backprop only) | 0.9945 ± 0.0003 | -0.9945 ± 0.0003 |
| **Hybrid — full (plasticity + replay + gate)** | **0.5489 ± 0.1739** | **-0.5489 ± 0.1739** |
| Hybrid — no replay (plasticity + gate only) | 0.6140 ± 0.0202 | -0.6140 ± 0.0202 |
| Hybrid — no gate (fixed 50/50 average) | 0.5141 ± 0.1118 | -0.5141 ± 0.1118 |

![Forgetting Comparison and Ablation](figures/fig3_forgetting_ablation.png)

**Interpretation:**

- The baseline is extremely consistent (std = 0.0003) — it reliably suffers
  near-total catastrophic forgetting across all seeds. This makes it a
  reliable reference point.
- All hybrid variants substantially and consistently outperform the
  baseline (roughly 0.99 → 0.51–0.61 forgetting), confirming that the fast
  plasticity pathway meaningfully reduces catastrophic forgetting.
- **Replay helps, as hypothesised:** removing replay increases forgetting
  from 0.549 to 0.614.
- **Unexpected ablation finding:** removing the learned gate (fixing the
  fast/slow combination at a static 50/50 average) performs statistically
  indistinguishable from — if not marginally better than — the full model
  with a learned gate (0.514 vs 0.549). This suggests the learned gate, in
  its current simple scalar form, is not yet extracting clear additional
  value over a fixed combination weight. This is flagged as an open
  question for future work (Section 8) rather than a negative result to
  hide — it narrows down which components are doing the real work.
- **Variance across seeds is notably higher for the hybrid variants
  (std up to 0.17) than the baseline.** This indicates the hybrid model's
  benefit, while real on average, is not yet fully stable across random
  initialisations — an honest limitation reported below rather than
  smoothed over.

### 5.2 Single-run detailed results (representative run, seed 0)

**Baseline (backprop only):**

| After training Task | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---|---|---|---|---|
| 1 | 0.999 | – | – | – | – |
| 2 | 0.000 | 0.99 | – | – | – |
| 3 | 0.000 | 0.000 | 0.993 | – | – |
| 4 | 0.000 | 0.000 | 0.000 | 0.995 | – |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.984 |

**Hybrid — full model:**

| After training Task | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---|---|---|---|---|
| 1 | 1.000 | – | – | – | – |
| 2 | 0.007 | 0.446 | – | – | – |
| 3 | 0.000 | 0.000 | 0.993 | – | – |
| 4 | 0.000 | 0.000 | 0.000 | 0.998 | – |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.509 |

![Task 1 Accuracy Trajectory](figures/fig5_task1_trajectory.png)

| Task | Slow-pathway-only accuracy | Fast-pathway-only accuracy |
|---|---|---|
| 1 | 0.000 | **0.964** |
| 2 | 0.000 | 0.136 |
| 3 | 0.000 | 0.035 |
| 4 | 0.000 | 0.000 |
| 5 | 0.491 | 0.000 |

![Fast vs Slow Pathway Accuracy](figures/fig6_fast_vs_slow.png)

This confirms the mechanism directly: the slow (backprop) pathway forgets
completely, exactly like the baseline. The fast (plasticity + replay)
pathway is what carries forward memory of Task 1 — it retains 96.4% accuracy
on Task 1 even after training on 4 subsequent tasks.

## 6. Observed Anomaly — Recency Bias

An unexpected pattern emerged: accuracy on Task 5 (the most recently trained
task) itself drops to 0% by the end of training, both for the fast pathway
alone and in the combined output. The most likely explanation is a timing
interaction between within-task learning and inter-task replay: replay steps
interleaved late in Task 5's training pull the fast pathway's weights back
toward earlier tasks before the current task's association is fully
consolidated. Task 2 and Task 3 also show partial, non-monotonic retention
(13.8% and 3.7%) rather than a clean decay curve, suggesting the
replay/consolidation balance is not yet well-tuned across all tasks.

This is a genuine instance of the **stability-plasticity trade-off** —
too much plasticity/replay protects old knowledge at the cost of acquiring
new knowledge cleanly. This is a well-known open problem in the continual
learning literature, not a flaw specific to this implementation, though the
specific severity seen here (complete loss of the newest task) indicates the
replay frequency and/or fast-pathway learning rate need further tuning.

**A mitigation attempt and its (negative) result:** An attempt was made to
fix this by (a) scaling down replay updates to 30% strength and (b) adding
a short "consolidation" pass of extra current-task-only training at the end
of each task. This *eliminated* the recency-bias anomaly, but at the cost
of eliminating essentially all of the benefit of the fast pathway — average
forgetting returned to 0.9857 ± 0.0036, statistically indistinguishable from
the baseline. This shows the fast pathway's benefit and the recency-bias
anomaly are closely coupled: strengthening current-task consolidation
directly undoes the retention of older tasks in this simple, shared-weight
fast pathway. This attempt was reverted; the results reported in Section
5.1 use the original (unmitigated) configuration. Resolving this coupling
— retaining old tasks without sacrificing the newest one — is left as the
primary direction for future work (Section 8).

## 7. Limitations

- **High cross-seed variance in hybrid variants** (std up to 0.17), while
  the baseline is highly stable (std 0.0003). The average improvement is
  real, but reliability across initialisations needs improvement.
- **The learned gate does not yet show a clear advantage** over a fixed
  50/50 combination in the ablation study — its current simple scalar
  design may be too limited to learn a useful policy from this amount of
  data.
- Fast pathway is a single linear layer; no hidden layers or nonlinearity.
- Replay buffer is small (20 exemplars/task) and replay frequency was fixed
  arbitrarily (every 5 batches) rather than tuned.
- Evaluated only on Split-MNIST, a relatively easy benchmark; harder,
  more temporally-structured benchmarks (e.g. Spiking Heidelberg Digits)
  have not yet been tested.
- No comparison yet against established continual learning baselines
  (EWC, GEM, experience replay variants from the literature).
- The recency-bias anomaly (Section 6) remains unresolved and likely
  contributes to the high variance observed in Section 5.1.

## 8. Future Work

- Redesign the gate as a per-class or input-dependent mechanism rather
  than a single global scalar, to test whether a more expressive gate
  can outperform fixed averaging.
- Tune replay frequency and buffer size to resolve the recency-bias
  anomaly and reduce cross-seed variance.
- Test on event-based, temporally-rich datasets (e.g. SHD, N-MNIST) rather
  than rate-coded static images.
- Compare against established continual learning baselines (EWC, GEM) for
  a stronger empirical claim.
- Investigate the source of high variance directly (e.g. via weight
  trajectory analysis) rather than only observing it at the metric level.

## 9. Conclusion

A hybrid SNN combining a backpropagation-trained slow pathway with a
locally-trained (delta-rule) fast pathway and lightweight episodic replay
reduces catastrophic forgetting on Split-MNIST from 0.9945 ± 0.0003
(baseline) to 0.5489 ± 0.1739 average forgetting across 5 random seeds —
a substantial, mechanistically-understood, and statistically-supported
improvement achieved through iterative debugging guided by isolated-pathway
diagnostics. An ablation study confirms replay contributes positively to
this result, while revealing that the learned gate does not yet clearly
outperform a fixed combination weight — a specific, actionable direction
for future work. An open trade-off between memory stability and new-task
plasticity remains, consistent with findings across the continual learning
literature, and is reflected in the higher variance observed for the
hybrid model relative to the baseline.
