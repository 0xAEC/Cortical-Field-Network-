# project-root/cfn/utils/pos_enc.py
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Literal

class PositionalEncodingBase(nn.Module):
    """Abstract base class for positional encoding modules."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds or concatenates positional encodings to the input tensor.
        Args:
            x (torch.Tensor): Input tensor, typically (B, C, H, W) or (B, N, C).
        Returns:
            torch.Tensor: Output tensor with positional encodings.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class AddSinePositionalEncoding2D(PositionalEncodingBase):
    """
    Adds 2D sinusoidal positional encodings to a 4D input tensor (B, C, H, W).
    Positional encodings are added to the channel dimension. The number of input
    channels C must be equal to `embedding_dim`.

    The `embedding_dim` is split:
    - Half for y-axis encodings (sine/cosine pairs).
    - Half for x-axis encodings (sine/cosine pairs).
    Requires `embedding_dim` to be divisible by 4.

    Args:
        embedding_dim (int): Number of channels C for the PE and input tensor.
        max_grid_shape (Tuple[int, int]): Max (Height, Width) for precomputation.
        temperature (float): Temperature for sine/cosine arguments.
        normalize_pe (bool): If True, L2 normalize the generated PE matrix.
        scale_factor_learnable (bool): If True, alpha is a learnable nn.Parameter.
                                       If False, alpha is a fixed float.
        initial_scale_factor (float): Initial value for alpha (if learnable) or
                                      the fixed scale factor (if not learnable).
        dropout_prob (float): Probability for dropout applied to the PE.
    """
    def __init__(self,
                 embedding_dim: int,
                 max_grid_shape: Tuple[int, int] = (64, 64),
                 temperature: float = 10000.0,
                 normalize_pe: bool = False,
                 scale_factor_learnable: bool = True,
                 initial_scale_factor: float = 1.0,
                 dropout_prob: float = 0.0):
        super().__init__(embedding_dim)

        if self.embedding_dim % 4 != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by 4 "
                "for 2D sinusoidal positional encoding using H/W split."
            )

        self.max_h, self.max_w = max_grid_shape
        self.temperature = temperature
        self.normalize_pe = normalize_pe
        self.scale_factor_learnable = scale_factor_learnable

        # Precompute positional encodings
        full_pe = self._build_sinusoidal_pe_2d()
        self.register_buffer('positional_encoding', full_pe, persistent=False)

        if self.scale_factor_learnable:
            self.alpha = nn.Parameter(torch.tensor(initial_scale_factor, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(initial_scale_factor, dtype=torch.float32), persistent=False)

        if dropout_prob > 0:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = nn.Identity()

    def _build_sinusoidal_pe_1d(self, length: int, num_channels_half_dim: int) -> torch.Tensor:
        """Helper to build 1D sinusoidal PE."""
        if num_channels_half_dim % 2 != 0:
             raise ValueError(f"num_channels_half_dim ({num_channels_half_dim}) must be even for sin/cos pairs.")
        
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1) # (length, 1)
        
        # Denominator for div_term needs to handle num_channels_half_dim being small (e.g., 2)
        # Original Transformer used d_model, here it's num_channels_half_dim
        # div_term_denominator = num_channels_half_dim for direct application.
        # For paired sin/cos over 'i' up to d_model/2, it's `num_channels_half_dim`.
        effective_dim_for_div = num_channels_half_dim
        
        div_term = torch.exp(
            torch.arange(0, num_channels_half_dim, 2, dtype=torch.float32) *
            -(math.log(self.temperature) / (effective_dim_for_div / 2.0)) # num_channels_half_dim/2 pairs
        ) # Shape (num_channels_half_dim / 2)
        
        pe_1d = torch.zeros(length, num_channels_half_dim)
        pe_1d[:, 0::2] = torch.sin(position * div_term)
        pe_1d[:, 1::2] = torch.cos(position * div_term)
        return pe_1d # (length, num_channels_half_dim)

    def _build_sinusoidal_pe_2d(self) -> torch.Tensor:
        """Builds and combines 2D sinusoidal PEs."""
        channels_per_dim = self.embedding_dim // 2 

        pe_y = self._build_sinusoidal_pe_1d(self.max_h, channels_per_dim) # (max_h, C/2)
        pe_x = self._build_sinusoidal_pe_1d(self.max_w, channels_per_dim) # (max_w, C/2)

        # pe_y: (max_h, C/2) -> (max_h, 1, C/2) -> (max_h, max_w, C/2)
        pe_y_expanded = pe_y.unsqueeze(1).repeat(1, self.max_w, 1)
        # pe_x: (max_w, C/2) -> (1, max_w, C/2) -> (max_h, max_w, C/2)
        pe_x_expanded = pe_x.unsqueeze(0).repeat(self.max_h, 1, 1)

        full_pe = torch.cat([pe_y_expanded, pe_x_expanded], dim=-1) # (max_h, max_w, C)
        full_pe = full_pe.permute(2, 0, 1).unsqueeze(0)             # (1, C, max_h, max_w)

        if self.normalize_pe:
            # Normalize across the channel dimension for each spatial location, or globally?
            # Global normalization might be too aggressive if max_grid_shape is large.
            # Per-location normalization is more like LayerNorm idea.
            # For now, global L2 norm of the PE matrix:
            norm = torch.linalg.norm(full_pe.view(-1))
            if norm > 1e-6: # Avoid division by zero
                 full_pe = full_pe / norm
            # Alternative: normalize per spatial position (pixel) over channels
            # norm = torch.linalg.norm(full_pe, dim=1, keepdim=True) # (1,1,max_h,max_w)
            # full_pe = full_pe / (norm + 1e-6)

        return full_pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds pre-computed positional encodings to the input tensor.
        Args:
            x (torch.Tensor): Input tensor (B, C, H, W). C must match self.embedding_dim.
        Returns:
            torch.Tensor: Output tensor with added positional encodings (B, C, H, W).
        """
        if x.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Input tensor channel dimension ({x.shape[1]}) "
                f"does not match PE embedding_dim ({self.embedding_dim})."
            )
        
        _, _, H, W = x.shape
        if H > self.max_h or W > self.max_w:
            # Option 1: Raise error (current)
            raise ValueError(
                f"Input grid shape ({H}, {W}) exceeds max_grid_shape ({self.max_h}, {self.max_w}). "
                "Re-initialize PositionalEncoding or ensure input is smaller."
            )
            # Option 2: Dynamically recompute/interpolate (more complex, not implemented here)
            # pe_to_add = self._build_sinusoidal_pe_2d_dynamic(H, W).to(x.device)

        # Crop the precomputed PE to the input's H, W
        pe_to_add = self.positional_encoding[:, :, :H, :W]
        
        # Apply dropout to PE before adding
        pe_to_add = self.dropout(pe_to_add)

        return x + self.alpha * pe_to_add # alpha broadcasts if scalar parameter or fixed buffer

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, max_grid_shape=({self.max_h}, {self.max_w}), "
            f"temperature={self.temperature}, normalize_pe={self.normalize_pe}, "
            f"scale_factor_learnable={self.scale_factor_learnable}, "
            f"dropout_prob={self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0}"
        )

# --- Commented out ideas for more complex PEs ---
"""
class LearnablePositionalEncoding2D(PositionalEncodingBase):
    '''
    Learns positional embeddings for a 2D grid.
    Concatenates or adds embeddings to the input tensor.

    Args:
        embedding_dim (int): Dimensionality of the positional embeddings.
        max_grid_shape (Tuple[int, int]): Maximum (Height, Width) of the grid.
        combination_mode (Literal['add', 'concat']): How to combine PE with input.
                                                   If 'concat', input channels C'
                                                   will become C' + embedding_dim.
    '''
    def __init__(self,
                 embedding_dim: int,
                 max_grid_shape: Tuple[int, int] = (32, 32),
                 combination_mode: Literal['add', 'concat'] = 'add'):
        super().__init__(embedding_dim)
        self.max_h, self.max_w = max_grid_shape
        self.combination_mode = combination_mode

        # Create separate learnable embeddings for H and W dimensions
        # This reduces parameters compared to a single (H*W, C) embedding table.
        self.row_embed = nn.Embedding(self.max_h, embedding_dim // (2 if combination_mode == 'add' else 1)) # Or full dim
        self.col_embed = nn.Embedding(self.max_w, embedding_dim // (2 if combination_mode == 'add' else 1)) # Or full dim
        
        # For 'add' mode, the input tensor channel dim must match embedding_dim
        # If splitting embedding_dim, ensure it's even.
        if combination_mode == 'add' and embedding_dim % 2 != 0:
            raise ValueError("If 'add' mode and splitting PE dim, embedding_dim must be even.")

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, H, W)
        B, C_in, H, W = x.shape
        
        if H > self.max_h or W > self.max_w:
            raise ValueError(f"Input grid shape ({H},{W}) > max_grid_shape ({self.max_h},{self.max_w})")

        i = torch.arange(W, device=x.device) # (W,)
        j = torch.arange(H, device=x.device) # (H,)

        x_emb = self.col_embed(i) # (W, D_col)
        y_emb = self.row_embed(j) # (H, D_row)

        # Expand to (H, W, D)
        # x_emb from (W, D_col) -> (1, W, D_col) -> (H, W, D_col)
        # y_emb from (H, D_row) -> (H, 1, D_row) -> (H, W, D_row)
        pos_x = x_emb.unsqueeze(0).repeat(H, 1, 1)
        pos_y = y_emb.unsqueeze(1).repeat(1, W, 1)

        if self.combination_mode == 'add':
            # Ensure C_in matches embedding_dim
            if C_in != self.embedding_dim:
                raise ValueError(f"For 'add' mode, input channels ({C_in}) must match "
                                 f"embedding_dim ({self.embedding_dim}).")
            # For 'add', we expect row_embed and col_embed to contribute to the full embedding_dim
            # D_col = D_row = embedding_dim / 2
            # Concatenate PEs from x and y, then reshape for adding
            pos_emb = torch.cat([pos_y, pos_x], dim=-1) # (H, W, D_row + D_col = D_emb)
            pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1) # (B, D_emb, H, W)
            return x + pos_emb
        elif self.combination_mode == 'concat':
            # D_col and D_row are usually smaller, or full embedding_dim if not split before.
            # Let's assume they form embedding_dim // 2 each for consistency here
            pos_emb = torch.cat([pos_y, pos_x], dim=-1) # (H, W, D_emb)
            pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1) # (B, D_emb, H, W)
            return torch.cat([x, pos_emb], dim=1) # Output channels: C_in + D_emb
        else:
            raise ValueError(f"Unknown combination_mode: {self.combination_mode}")
            
    def extra_repr(self) -> str:
        return (f"embedding_dim={self.embedding_dim}, max_grid_shape=({self.max_h}, {self.max_w}), "
                f"combination_mode='{self.combination_mode}'")

def get_positional_encoder(name: str, config: dict) -> PositionalEncodingBase:
    if name.lower() == 'sine2d_add':
        return AddSinePositionalEncoding2D(**config)
    # elif name.lower() == 'learnable2d':
    #     return LearnablePositionalEncoding2D(**config)
    else:
        raise ValueError(f"Unknown positional encoder name: {name}")

"""

if __name__ == '__main__':
    print("--- Positional Encoding Tests ---")
    B, C, H, W = 2, 16, 8, 10 # C needs to be divisible by 4 for AddSine2D
    dummy_input = torch.randn(B, C, H, W)

    # Test AddSinePositionalEncoding2D
    print("\nTesting AddSinePositionalEncoding2D:")
    sine_pe_config = {
        'embedding_dim': C,
        'max_grid_shape': (16, 16), # Larger than input H, W
        'temperature': 1000.0,
        'normalize_pe': True,
        'scale_factor_learnable': True,
        'initial_scale_factor': 0.5,
        'dropout_prob': 0.1
    }
    sine_pe_module = AddSinePositionalEncoding2D(**sine_pe_config)
    print(sine_pe_module)
    
    output_sine = sine_pe_module(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape (sine):", output_sine.shape)
    assert output_sine.shape == dummy_input.shape
    print(f"Learnable alpha: {sine_pe_module.alpha.item() if sine_pe_module.scale_factor_learnable else sine_pe_module.alpha}")

    # Test with smaller grid to check PE cropping
    dummy_input_small = torch.randn(B, C, 4, 5)
    output_sine_small = sine_pe_module(dummy_input_small)
    print("Input shape small:", dummy_input_small.shape)
    print("Output shape (sine) small:", output_sine_small.shape)
    assert output_sine_small.shape == dummy_input_small.shape

    try:
        dummy_input_too_large = torch.randn(B,C, 20, 20) # Larger than max_grid_shape
        sine_pe_module(dummy_input_too_large)
    except ValueError as e:
        print(f"Caught expected error for too large input: {e}")

    # --- Example for Learnable PE (if uncommented) ---
    # print("\nTesting LearnablePositionalEncoding2D (add mode):")
    # learnable_pe_add_config = {
    #     'embedding_dim': C, # For add mode, this must match input C
    #     'max_grid_shape': (16, 16),
    #     'combination_mode': 'add'
    # }
    # learnable_pe_add_module = LearnablePositionalEncoding2D(**learnable_pe_add_config)
    # print(learnable_pe_add_module)
    # output_learnable_add = learnable_pe_add_module(dummy_input)
    # print("Output shape (learnable add):", output_learnable_add.shape)
    # assert output_learnable_add.shape == dummy_input.shape

    # print("\nTesting LearnablePositionalEncoding2D (concat mode):")
    # learnable_pe_concat_config = {
    #     'embedding_dim': 8, # PE channels to concatenate
    #     'max_grid_shape': (16, 16),
    #     'combination_mode': 'concat'
    # }
    # learnable_pe_concat_module = LearnablePositionalEncoding2D(**learnable_pe_concat_config)
    # print(learnable_pe_concat_module)
    # output_learnable_concat = learnable_pe_concat_module(dummy_input)
    # print("Output shape (learnable concat):", output_learnable_concat.shape)
    # assert output_learnable_concat.shape == (B, C + learnable_pe_concat_config['embedding_dim'], H, W)
    
    print("\nPositional encoding tests seem fine.")
