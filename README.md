# Hybrid Plasticity Spiking Neural Networks for Continual Learning

A Spiking Neural Network that combines a backpropagation-trained pathway
with a local, biologically-inspired plasticity pathway and episodic
replay, aimed at reducing catastrophic forgetting in continual learning.

![Architecture](figures/fig1_architecture.png)

## Overview

Spiking Neural Networks trained purely by backpropagation-through-time
suffer from catastrophic forgetting when trained sequentially on new
tasks. This project investigates whether a small, locally-trained "fast"
synaptic pathway, combined with lightweight episodic replay, can reduce
this forgetting — inspired by Complementary Learning Systems theory
(McClelland et al., 1995), in which the hippocampus learns quickly via
local plasticity while the cortex consolidates knowledge slowly.

**Headline result:** on the Split-MNIST class-incremental benchmark
(5 sequential tasks, 5 random seeds), average forgetting is reduced from
**0.9945 ± 0.0003** (pure-backpropagation baseline) to **0.5489 ± 0.1739**
with the hybrid model.

See [`REPORT.md`](REPORT.md) for the full write-up, including the
development process, ablation study, and honestly-reported limitations.

## Architecture

- **Slow pathway** — two layers of Leaky Integrate-and-Fire (LIF) neurons
  trained end-to-end via backpropagation with a surrogate gradient.
- **Fast pathway** — a single linear layer updated only by a local delta
  rule (no backpropagation, no gradient through time).
- **Gate** — a learned scalar combining the two pathways' outputs.
- **Episodic replay buffer** — after each task, a small number of
  exemplars are stored and periodically replayed through the fast
  pathway only, protecting old associations without touching the slow
  pathway.

## Repository Structure

```
.
├── lif_neuron.py          # Slow pathway: LIF neuron layer (surrogate gradient)
├── stdp.py                 # Fast pathway: local delta-rule plasticity
├── hybrid_model.py         # Combines both pathways with a learned gate
├── baseline_model.py       # Control: pure-backpropagation SNN
├── data_utils.py           # Split-MNIST class-incremental data loading
├── train.py                # Main training + evaluation script
├── run_experiments.py      # Multi-seed ablation study
├── generate_figures.py     # Reproduces all figures from the results
├── figures/                # Generated figures
├── REPORT.md                # Full technical report
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run a single baseline-vs-hybrid comparison:

```bash
python train.py
```

Run the full multi-seed ablation study (baseline, hybrid, and two
ablated variants, 5 seeds each — this performs 20 full training runs
and may take 30–60+ minutes on a free-tier GPU):

```bash
python run_experiments.py
```

Regenerate the figures from the reported results:

```bash
python generate_figures.py
```

## Results

| Variant | Average Forgetting (mean ± std, 5 seeds) |
|---|---|
| Baseline (backpropagation only) | 0.9945 ± 0.0003 |
| **Hybrid (full: plasticity + replay + gate)** | **0.5489 ± 0.1739** |
| Hybrid (no replay) | 0.6140 ± 0.0202 |
| Hybrid (no gate) | 0.5141 ± 0.1118 |

![Forgetting comparison](figures/fig3_forgetting_ablation.png)

Full results, per-task breakdowns, isolated-pathway diagnostics, and a
discussion of an observed stability-plasticity trade-off (recency bias)
are in [`REPORT.md`](REPORT.md).

## Limitations

This is a small-scale exploratory study, not a publication-ready paper.
Notable open items (detailed in `REPORT.md`):
- Evaluated only on Split-MNIST, a relatively easy benchmark.
- No comparison yet against established continual learning baselines
  (EWC, GEM, etc.).
- The learned gate does not yet show a clear advantage over a fixed
  combination weight.
- Cross-seed variance in the hybrid model is notably higher than the
  baseline.

## References

- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995).
  Why there are complementary learning systems in the hippocampus and
  neocortex. *Psychological Review*.
- Zenke, F., & Vogels, T. P. (2021). The remarkable robustness of
  surrogate gradient learning for instilling complex function in
  spiking neural networks. *Neural Computation*.
- Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for
  continual learning. *NeurIPS*.

## License

MIT License — see [`LICENSE`](LICENSE).
