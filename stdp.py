"""
stdp.py

Implements the "fast pathway" component: a linear associative memory
updated by a local, biologically-plausible learning rule instead of
backpropagation.

Design history (see REPORT.md for the full account):
  1. Unsupervised Hebbian STDP -- had no notion of class labels and
     provided no benefit for continual learning.
  2. Teacher-guided STDP with separate potentiation/depression traces --
     the two terms cancelled each other out, again providing no benefit.
  3. A self-referential spike threshold made the readout unstable.
  4. Final design (below): a local delta rule (Widrow-Hoff), which is
     stable, well understood, and still fully local (no backpropagation,
     no gradient through time) -- consistent with "three-factor" local
     learning rules used in the reward-modulated STDP / e-prop literature.
"""

import torch
import torch.nn as nn


class STDPLinear(nn.Module):
    """
    A linear layer whose weights are updated by a local delta rule:

        weight += scale * lr * (input) x (target - prediction)

    No optimizer or gradient tape ever touches these weights.
    """

    def __init__(self, in_features, out_features, lr=0.05, w_max=2.0):
        super().__init__()
        self.weight = torch.zeros(out_features, in_features)
        self.lr = lr
        self.w_max = w_max

    def to_device(self, device):
        self.weight = self.weight.to(device)

    def forward(self, x):
        """x: (B, in_features) -> (B, out_features)"""
        return x @ self.weight.T

    @torch.no_grad()
    def update(self, x, teacher_onehot, scale=1.0):
        """
        Local delta-rule update.

        Args:
            x: (B, in_features) -- presynaptic activity (e.g. average
               firing rate).
            teacher_onehot: (B, out_features) -- correct label, one-hot.
            scale: multiplier on the update magnitude. Used to make replay
               updates gentler than current-task updates when desired.
        """
        prediction = self.forward(x)
        error = teacher_onehot - prediction
        dw = scale * self.lr * torch.einsum('bi,bj->ji', x, error) / x.shape[0]
        self.weight = torch.clamp(self.weight + dw, min=-self.w_max, max=self.w_max)
