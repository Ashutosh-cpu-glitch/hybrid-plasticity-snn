"""
baseline_model.py

Control condition: a standard SNN trained purely by backpropagation,
with no local plasticity and no replay. Used to measure catastrophic
forgetting in the absence of any mitigation.
"""

import torch.nn as nn
from lif_neuron import LIFLayer


class BaselineSNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,
                 beta=0.9, threshold=1.0):
        super().__init__()
        self.layer1 = LIFLayer(in_features, hidden_features, beta, threshold)
        self.layer2 = LIFLayer(hidden_features, out_features, beta, threshold)

    def forward(self, x_seq):
        hidden_spikes = self.layer1(x_seq)
        out_spikes = self.layer2(hidden_spikes)
        return out_spikes.sum(dim=0)  # spike count = output "rate"
