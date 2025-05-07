import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional, Literal, Tuple # Added Tuple

class PixelwiseLayerNorm(nn.Module):
    """
    Applies LayerNorm over the channel dimension for each spatial location (pixel).

    This is equivalent to nn.LayerNorm if the input tensor is permuted
    so that the channel dimension is the last dimension.

    Input shape: (B, C, H, W)
    Output shape: (B, C, H, W)
    """
    def __init__(self,
                 num_channels: int,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 device=None, # Added for completeness, though parent module .to() handles it
                 dtype=None): # Added for completeness
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # nn.LayerNorm expects normalized_shape to be the last dimension(s)
        self.norm = nn.LayerNorm(normalized_shape=num_channels,
                                 eps=eps,
                                 elementwise_affine=elementwise_affine,
                                 device=device,
                                 dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Input tensor channel dimension ({x.shape[1]}) "
                             f"does not match num_channels ({self.num_channels})")
        # x: (B, C, H, W) -> (B, H, W, C) for LayerNorm
        x_permuted = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_permuted)
        # x_norm: (B, H, W, C) -> (B, C, H, W) back to original layout
        return x_norm.permute(0, 3, 1, 2)

    def extra_repr(self) -> str:
        return f"{self.num_channels}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class CFNCell(nn.Module):
    """
    Core Cortical Field Network (CFN) recurrent update cell, as per CFN 2.0.
    This cell implements a specific form of the local update rule f_θ, inspired
    by gated recurrent units and related to discretized Amari-type neural fields.

    The update rule for the field state H at each cell location v is:
        h_v(t+1) = g_v(t) ⊙ h_v(t) + (1 - g_v(t)) ⊙ h̃_v(t+1)
    where:
        h̃_v(t+1) = tanh(norm_u(conv_u(h_v(t), {h_u(t): u ∈ N(v)})))  (candidate state)
        g_v(t)   = sigmoid(norm_g(conv_g(h_v(t), {h_u(t): u ∈ N(v)})))    (update gate)

    This formulation is analogous to the GRU update and aligns with Section 3.1(C)
    of the CFN 2.0 paper. The dynamics can also be seen as a discretized form of
    Amari-type neural fields (Section 4.1).

    Args:
        channels (int): Number of feature channels C in the field state H.
        kernel_size (int): Size of the convolutional kernel (must be a positive odd integer).
        padding_mode (Literal): Padding mode for convolutions.
            Options: 'zeros', 'reflect', 'replicate', 'circular'.
        normalization (Optional[Literal['group', 'layer']]): Type of normalization to apply.
            'layer': PixelwiseLayerNorm (LayerNorm across channels per spatial location).
                     This aligns with "Layer normalization ... applied across channels
                     within each cell state" (CFN 2.0 Paper, Sec. 5).
            'group': GroupNorm. `group_norm_num_groups` controls the number of groups.
                     If `group_norm_num_groups=1` (default), normalizes over all channels
                     and spatial dimensions combined, per batch item.
            None: No normalization.
        group_norm_num_groups (int): Number of groups for GroupNorm if `normalization == 'group'`.
            Default is 1, which normalizes all channels and spatial dims together.
            A common setting is a value like 8, 16, 32, or `channels` (for InstanceNorm).
        bias (bool): Whether to include bias terms in the convolutional layers.
        gate_bias_init (Optional[float]): If provided, initializes the bias of the gate
            convolution `conv_g` to this value. A positive value (e.g., 2.0) biases
            the gate `g` towards 1 (open) initially, promoting identity-like updates
            at the beginning of training, as suggested in CFN 2.0 Paper (Sec. 5).
            If None, bias is initialized to zero.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'replicate',
        normalization: Optional[Literal['group', 'layer']] = 'layer', # Defaulted to 'layer' as per paper's suggestion
        group_norm_num_groups: int = 1, # Used only if normalization == 'group'
        bias: bool = True,
        gate_bias_init: Optional[float] = 2.0, # Biasing gate towards 1 initially
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError("Number of channels must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.normalization_type = normalization
        self.group_norm_num_groups = group_norm_num_groups
        self.bias = bias
        self.gate_bias_init = gate_bias_init

        padding = kernel_size // 2

        # Convolution for the candidate state update path
        self.conv_u = nn.Conv2d(channels, channels, kernel_size,
                                padding=padding,
                                padding_mode=padding_mode,
                                bias=bias)
        # Convolution for the gate path
        self.conv_g = nn.Conv2d(channels, channels, kernel_size,
                                padding=padding,
                                padding_mode=padding_mode,
                                bias=bias)

        # Normalization layers
        if normalization == 'group':
            if channels % group_norm_num_groups != 0:
                raise ValueError(f"Number of channels ({channels}) must be divisible by "
                                 f"group_norm_num_groups ({group_norm_num_groups}).")
            self.norm_u = nn.GroupNorm(group_norm_num_groups, channels, affine=True)
            self.norm_g = nn.GroupNorm(group_norm_num_groups, channels, affine=True)
        elif normalization == 'layer':
            self.norm_u = PixelwiseLayerNorm(channels, elementwise_affine=True)
            self.norm_g = PixelwiseLayerNorm(channels, elementwise_affine=True)
        elif normalization is None:
            self.norm_u = nn.Identity()
            self.norm_g = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type: {normalization}. "
                             "Choose from 'group', 'layer', or None.")

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reinitialize model parameters according to common practices and paper suggestions."""
        # Initialize conv weights for candidate state path (followed by tanh)
        init.kaiming_normal_(self.conv_u.weight, mode='fan_in', nonlinearity='tanh')
        if self.conv_u.bias is not None:
            init.zeros_(self.conv_u.bias)

        # Initialize conv weights for gate path (followed by sigmoid)
        init.kaiming_normal_(self.conv_g.weight, mode='fan_in', nonlinearity='sigmoid')
        if self.conv_g.bias is not None:
            if self.gate_bias_init is not None:
                # Bias the gate towards being open (g -> 1) initially
                # sigmoid(2.0) approx 0.88; sigmoid(5.0) approx 0.99
                init.constant_(self.conv_g.bias, self.gate_bias_init)
            else:
                init.zeros_(self.conv_g.bias)

        # Affine parameters in Norm layers (gamma, beta) are typically initialized
        # to 1 and 0 respectively by PyTorch's nn.LayerNorm/nn.GroupNorm.

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Performs one recurrent update step of the CFN.

        Args:
            H (torch.Tensor): The current field state tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: The next field state tensor H(t+1) of shape (B, C, H, W).
        """
        # Candidate state path
        # h̃_v(t+1) = tanh(norm_u(conv_u(H)))
        candidate_state = torch.tanh(self.norm_u(self.conv_u(H)))

        # Gate path
        # g_v(t) = sigmoid(norm_g(conv_g(H)))
        update_gate = torch.sigmoid(self.norm_g(self.conv_g(H)))

        # Gated update: h_v(t+1) = g_v(t) * H + (1 - g_v(t)) * h̃_v(t+1)
        # Using '*' for element-wise multiplication (Hadamard product ⊙)
        H_next = update_gate * H + (1 - update_gate) * candidate_state

        return H_next

    def extra_repr(self) -> str:
        norm_repr = (
            f'GroupNorm(num_groups={self.group_norm_num_groups})' if isinstance(self.norm_u, nn.GroupNorm)
            else 'PixelwiseLayerNorm' if isinstance(self.norm_u, PixelwiseLayerNorm)
            else 'None' if isinstance(self.norm_u, nn.Identity)
            else str(type(self.norm_u).__name__)
        )
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"padding_mode='{self.padding_mode}', normalization={norm_repr}, "
            f"bias={self.bias is not None and self.conv_u.bias is not None}, " # check if bias was enabled and created
            f"gate_bias_init={self.gate_bias_init}"
        )
