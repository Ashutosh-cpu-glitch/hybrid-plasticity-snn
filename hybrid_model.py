"""
hybrid_model.py

Combines two complementary learning pathways, inspired by Complementary
Learning Systems theory (McClelland et al., 1995):

    SLOW PATHWAY  -> LIFLayer (lif_neuron.py)  -- backpropagation-trained
    FAST PATHWAY  -> STDPLinear (stdp.py)       -- local delta-rule plasticity

A learned scalar gate combines the two pathways' outputs. An episodic
replay buffer periodically re-exposes the fast pathway to a small number
of examples from previous tasks (hippocampal-replay-inspired), without
affecting the slow pathway.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lif_neuron import LIFLayer
from stdp import STDPLinear


class HybridSNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,
                 beta=0.9, threshold=1.0, use_gate=True):
        super().__init__()

        # Slow pathway: standard backpropagation-trained SNN
        self.slow_layer1 = LIFLayer(in_features, hidden_features, beta, threshold)
        self.slow_layer2 = LIFLayer(hidden_features, out_features, beta, threshold)

        # Fast pathway: delta-rule trained, direct input -> output projection
        self.fast_layer = STDPLinear(in_features, out_features)

        # Gate: learns how much weight to give the fast vs. slow pathway.
        # use_gate=False -> ablation mode: fixed 50/50 average.
        self.use_gate = use_gate
        self.gate = nn.Parameter(torch.tensor(0.5))

    def reset_fast_pathway(self, batch_size, device):
        """Ensures the fast pathway's weights are on the correct device."""
        self.fast_layer.to_device(device)

    def _pre_rate(self, x_seq):
        """Average firing rate of a spike train: (T, B, F) -> (B, F)."""
        return x_seq.mean(dim=0)

    def fast_pathway_replay(self, x_seq, labels, num_classes=10, scale=1.0):
        """
        Replay: re-exposes the fast pathway to stored examples from
        previous tasks, so it does not lose those associations while
        learning a new task. The slow pathway is not affected.
        """
        pre_rate = self._pre_rate(x_seq)
        teacher_onehot = F.one_hot(labels, num_classes=num_classes).float().to(x_seq.device)
        self.fast_layer.update(pre_rate, teacher_onehot, scale=scale)

    def forward_debug(self, x_seq):
        """
        Diagnostic only: returns the slow and fast pathway outputs
        separately (uncombined), to analyse which pathway retains or
        forgets a given task. Performs no plasticity update.
        """
        slow_hidden_spikes = self.slow_layer1(x_seq)
        slow_out_spikes = self.slow_layer2(slow_hidden_spikes)
        slow_output = slow_out_spikes.sum(dim=0)

        pre_rate = self._pre_rate(x_seq)
        fast_output = self.fast_layer(pre_rate)

        return slow_output, fast_output

    def forward(self, x_seq, update_plasticity=True, labels=None, num_classes=10):
        """
        Args:
            x_seq: (T, B, in_features) spike train input.
            labels: (B,) integer class labels, required during training
                (used as the fast pathway's local teaching signal).
            update_plasticity: if True, applies a fast-pathway delta-rule
                update using `labels`.

        Returns:
            (B, out_features) combined output.
        """
        device = x_seq.device

        # Slow pathway (backpropagation)
        slow_hidden_spikes = self.slow_layer1(x_seq)
        slow_out_spikes = self.slow_layer2(slow_hidden_spikes)
        slow_output = slow_out_spikes.sum(dim=0)

        # Fast pathway (local delta rule, no backpropagation)
        pre_rate = self._pre_rate(x_seq)
        fast_output = self.fast_layer(pre_rate)

        if update_plasticity and labels is not None:
            teacher_onehot = F.one_hot(labels, num_classes=num_classes).float().to(device)
            self.fast_layer.update(pre_rate, teacher_onehot)

        # Combine via gate (or fixed 50/50 average in the ablation setting)
        alpha = torch.sigmoid(self.gate) if self.use_gate else 0.5
        combined = alpha * fast_output + (1 - alpha) * slow_output
        return combined
