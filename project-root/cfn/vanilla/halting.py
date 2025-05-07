# project-root/cfn/vanilla/halting.py

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional

class AdaptiveHaltingUnit(nn.Module):
    """
    Adaptive Halting Unit (g_φ) for CFN, inspired by PonderNet.
    This module predicts the probability of halting at the current step.

    As per CFN 2.0 Paper (Sec. 3.1 D):
        p_halt_v(t) = σ(u_φ(h_v(t)))
    where u_φ is a small neural network.

    In PonderNet, this corresponds to calculating the per-step halting
    probability lambda_n (referred to as p_k or p_t here for clarity
    at step k or t).

    The module u_φ is typically a simple linear projection from the
    field state H(t) to a scalar logit per spatial location, followed by a sigmoid.

    Args:
        input_channels (int): Number of channels in the input field state H(t).
        halting_bias_init (float): Initial bias for the final linear layer
            projecting to halting logits. A negative value (e.g., -3.0)
            encourages the network to take more steps initially by making
            p_halt small. (sigmoid(-3.0) approx 0.047).
        kernel_size (int): Kernel size for the convolutional layer if u_phi is conv based.
                           Defaults to 1 for pixel-wise processing. Must be odd.
    """
    def __init__(self,
                 input_channels: int,
                 halting_bias_init: float = -3.0,
                 kernel_size: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.halting_bias_init = halting_bias_init
        self.kernel_size = kernel_size

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer for u_phi.")

        padding = kernel_size // 2

        # u_φ: A small network to predict halting logits.
        # Simplest form: a 1x1 convolution to map C channels to 1 logit channel.
        # The paper mentions "a small network". A single Conv2D is a common choice.
        self.u_phi = nn.Conv2d(
            in_channels=input_channels,
            out_channels=1, # Output one logit per spatial location
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters for the halting unit."""
        # Initialize weights of u_φ
        # He init or Kaiming init is common for conv layers.
        # Since it's followed by sigmoid eventually, 'linear' might be more neutral,
        # but Kaiming with a small gain or Xavier uniform is also fine.
        init.xavier_uniform_(self.u_phi.weight, gain=init.calculate_gain('sigmoid')) # or 'linear' gain

        # Initialize bias of u_φ
        # PonderNet suggests initializing bias such that initial p_halt is small.
        if self.u_phi.bias is not None:
            init.constant_(self.u_phi.bias, self.halting_bias_init)

    def forward(self, H_t: torch.Tensor) -> torch.Tensor:
        """
        Compute the per-pixel halting probability for the current step.

        Args:
            H_t (torch.Tensor): The current field state, shape (B, C, H, W).

        Returns:
            torch.Tensor: Per-pixel halting probabilities p_halt(t), shape (B, 1, H, W).
                          Values are in [0, 1].
        """
        if H_t.shape[1] != self.input_channels:
            raise ValueError(
                f"Input tensor H_t has {H_t.shape[1]} channels, "
                f"but AdaptiveHaltingUnit was initialized with {self.input_channels} channels."
            )

        # Get logits from u_φ
        halting_logits = self.u_phi(H_t)  # Shape: (B, 1, H, W)

        # Apply sigmoid to get probabilities
        p_halt_t = torch.sigmoid(halting_logits) # Shape: (B, 1, H, W)

        return p_halt_t

    def extra_repr(self) -> str:
        return (
            f"input_channels={self.input_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"halting_bias_init={self.halting_bias_init}"
        )

if __name__ == '__main__':
    # Example Usage:
    batch_size, channels, height, width = 2, 16, 32, 32
    
    # 1. Test with default kernel_size=1
    halting_unit_k1 = AdaptiveHaltingUnit(input_channels=channels)
    print("Halting Unit (kernel_size=1):", halting_unit_k1)
    
    dummy_H_t_k1 = torch.randn(batch_size, channels, height, width)
    p_halt_t_k1 = halting_unit_k1(dummy_H_t_k1)
    
    print("Input H_t shape (k=1):", dummy_H_t_k1.shape)
    print("Output p_halt_t shape (k=1):", p_halt_t_k1.shape)
    print("Output p_halt_t min/max (k=1):", p_halt_t_k1.min().item(), p_halt_t_k1.max().item())
    assert p_halt_t_k1.shape == (batch_size, 1, height, width)
    assert p_halt_t_k1.min() >= 0.0 and p_halt_t_k1.max() <= 1.0

    # 2. Test with kernel_size=3
    halting_unit_k3 = AdaptiveHaltingUnit(input_channels=channels, kernel_size=3, halting_bias_init=-2.0)
    print("\nHalting Unit (kernel_size=3):", halting_unit_k3)
    
    dummy_H_t_k3 = torch.randn(batch_size, channels, height, width)
    p_halt_t_k3 = halting_unit_k3(dummy_H_t_k3)
    
    print("Input H_t shape (k=3):", dummy_H_t_k3.shape)
    print("Output p_halt_t shape (k=3):", p_halt_t_k3.shape)
    print("Output p_halt_t min/max (k=3):", p_halt_t_k3.min().item(), p_halt_t_k3.max().item())
    assert p_halt_t_k3.shape == (batch_size, 1, height, width)
    assert p_halt_t_k3.min() >= 0.0 and p_halt_t_k3.max() <= 1.0

    print("\nExample tests passed!") 

## note to self: This AdaptiveHaltingUnit only computes p_halt_t. It does not implement the full PonderNet logic (e.g., sampling N, calculating cumulative probabilities, ponder cost).
## That more complex logic will reside in the main VanillaCFN model in model.py, which will use this AdaptiveHaltingUnit at each recurrent step.
