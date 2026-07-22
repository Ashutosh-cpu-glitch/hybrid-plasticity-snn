"""
lif_neuron.py

Implements the "slow pathway" component: a Leaky Integrate-and-Fire (LIF)
spiking neuron layer trained end-to-end via backpropagation using a
surrogate gradient, since spikes are non-differentiable.
"""

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    """
    A spike is a hard step function (fire / no fire), which has zero
    gradient almost everywhere, making standard backpropagation
    ineffective. This custom autograd function keeps the true step
    function in the forward pass, but substitutes a smooth surrogate
    (fast sigmoid) in the backward pass so that gradients can flow
    (Zenke & Vogels, 2021).
    """

    @staticmethod
    def forward(ctx, membrane_minus_threshold):
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = 5.0
        surrogate_grad = 1.0 / (1.0 + alpha * x.abs()) ** 2
        return grad_output * surrogate_grad


spike_fn = SurrogateSpike.apply


class LIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons.

    At each timestep:
      1. An input current arrives (weighted sum of the previous layer's
         spikes).
      2. The membrane potential leaks slightly and integrates the new
         input current.
      3. If the membrane potential crosses the threshold, the neuron
         fires a spike.
      4. After firing, the membrane potential is reset.
    """

    def __init__(self, in_features, out_features, beta=0.9, threshold=1.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.beta = beta            # membrane decay (leak) factor
        self.threshold = threshold  # spike threshold

    def forward(self, x_seq):
        """
        Args:
            x_seq: (T, B, in_features) spike train input over T timesteps.

        Returns:
            spikes: (T, B, out_features)
        """
        T, B, _ = x_seq.shape
        mem = torch.zeros(B, self.fc.out_features, device=x_seq.device)
        spikes = []

        for t in range(T):
            current = self.fc(x_seq[t])
            mem = self.beta * mem + current           # leaky integration
            spk = spike_fn(mem - self.threshold)       # fire or not
            mem = mem - spk * self.threshold             # soft reset
            spikes.append(spk)

        return torch.stack(spikes, dim=0)  # (T, B, out_features)
