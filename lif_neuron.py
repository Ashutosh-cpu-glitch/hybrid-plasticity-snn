"""
lif_neuron.py

Implements the "slow pathway" component: a Leaky Integrate-and-Fire (LIF)
spiking neuron layer trained end-to-end via backpropagation using a
surrogate gradient, since spikes are non-differentiable.
"""

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
  

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
   
    def __init__(self, in_features, out_features, beta=0.9, threshold=1.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.beta = beta            # membrane decay (leak) factor
        self.threshold = threshold  # spike threshold

    def forward(self, x_seq):
        T, B, _ = x_seq.shape
        mem = torch.zeros(B, self.fc.out_features, device=x_seq.device)
        spikes = []

        for t in range(T):
            current = self.fc(x_seq[t])
            mem = self.beta * mem + current           
            spk = spike_fn(mem - self.threshold)       
            mem = mem - spk * self.threshold             
            spikes.append(spk)

        return torch.stack(spikes, dim=0)  # (T, B, out_features)
