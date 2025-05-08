# project-root/cfn/vanilla/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable, Tuple, Union, Literal # Make sure Literal is imported

# Correct relative import needed if model.py is in cfn/vanilla/
# This tries to import AddSinePositionalEncoding2D from ../utils/pos_enc.py
try:
    from ..utils.pos_enc import AddSinePositionalEncoding2D as AddSinePositionalEncoding
except ImportError:
    # Fallback dummy class if the import fails
    class AddSinePositionalEncoding(nn.Module):
        def __init__(self, embedding_dim: int, max_grid_shape: Tuple[int, int] = (32,32), **kwargs):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.max_grid_shape = max_grid_shape
            print(f"Warning: cfn.utils.pos_enc.AddSinePositionalEncoding not found or failed to import. Using dummy Identity for PE. Expected embedding_dim={embedding_dim}, max_grid_shape={max_grid_shape}")
        def forward(self, x): return x
        def extra_repr(self) -> str: return "Dummy Identity for Positional Encoding"

# Imports from the same directory level (vanilla)
from .cell import CFNCell
from .halting import AdaptiveHaltingUnit


class VanillaCFN(nn.Module):
    """
    Vanilla Cortical Field Network (CFN) with PonderNet-style adaptive computation.
    Implements key ideas from CFN 2.0 paper and PonderNet (Banino et al., 2021).

    Args:
        input_raw_channels (int): Channels in the raw input tensor x.
        output_raw_channels (int): Channels in the final task output.
        field_channels (int): Channels C in the internal field state H.
        cfn_cell_kwargs (Dict): Keyword arguments for CFNCell (REQUIRED).
        halting_unit_kwargs (Dict): Keyword arguments for AdaptiveHaltingUnit (REQUIRED).
        grid_shape (Optional[Tuple[int, int]]): Expected (H, W) of grid. Required if use_pos_enc=True. Default: None.
        max_steps (int): Maximum recurrent steps (N_max in PonderNet). Default: 10.
        ponder_reg_beta (float): Weight for the ponder cost (L_P regularization). Default: 0.01.
        use_positional_encoding (bool): If True, add positional encodings to H_0. Default: True.
        pos_enc_kwargs (Optional[Dict]): Kwargs for AddSinePositionalEncoding. Default: None.
                                         'embedding_dim' is field_channels,
                                         'max_grid_shape' is grid_shape.
        input_encoder (Optional[nn.Module]): Encodes raw input to H_0. Default: Conv2D(1x1).
        output_decoder (Optional[nn.Module]): Decodes H_t to task output. Default: Conv2D(1x1).
        eps_stability (float): Small epsilon for numerical stability. Default: 1e-8.
        inference_mode (Literal['expected', 'sample_batchwise']): Strategy for eval. Default: 'expected'.
    """
    def __init__(
        self,
        # Non-default arguments FIRST - Corrected Order
        input_raw_channels: int,
        output_raw_channels: int,
        field_channels: int,
        cfn_cell_kwargs: Dict,
        halting_unit_kwargs: Dict,
        # Optional arguments with default values LAST - Corrected Order
        grid_shape: Optional[Tuple[int, int]] = None, # H, W
        max_steps: int = 10,
        ponder_reg_beta: float = 0.01,
        use_positional_encoding: bool = True,
        pos_enc_kwargs: Optional[Dict] = None,
        input_encoder: Optional[nn.Module] = None,
        output_decoder: Optional[nn.Module] = None,
        eps_stability: float = 1e-8,
        inference_mode: Literal['expected', 'sample_batchwise'] = 'expected'
    ):
        super().__init__()

        # --- Basic argument validation ---
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        # Grid_shape check moved down to where it's used

        # --- Store configuration ---
        self.input_raw_channels = input_raw_channels
        self.output_raw_channels = output_raw_channels
        self.field_channels = field_channels
        # Store the kwargs dictionaries as they are needed to instantiate modules
        self.cfn_cell_kwargs_config = cfn_cell_kwargs
        self.halting_unit_kwargs_config = halting_unit_kwargs
        self.grid_shape = grid_shape
        self.max_steps = max_steps
        self.ponder_reg_beta = ponder_reg_beta
        self.use_positional_encoding = use_positional_encoding
        self.eps_stability = eps_stability
        self.inference_mode = inference_mode

        # --- Input Encoder ---
        if input_encoder is None:
            self.input_encoder = nn.Conv2d(self.input_raw_channels, self.field_channels, kernel_size=1)
            # Simple initialization
            nn.init.kaiming_normal_(self.input_encoder.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if self.input_encoder.bias is not None: nn.init.zeros_(self.input_encoder.bias)
        else:
            # Use the provided encoder module
            self.input_encoder = input_encoder

        # --- Positional Encoding ---
        if self.use_positional_encoding:
            if self.grid_shape is None: # Perform check here
                raise ValueError("grid_shape must be provided if use_positional_encoding is True.")
            _pe_kwargs = pos_enc_kwargs if pos_enc_kwargs is not None else {}
            self.positional_encoder = AddSinePositionalEncoding(
                embedding_dim=self.field_channels,
                max_grid_shape=self.grid_shape,
                **_pe_kwargs
            )
        else:
            self.positional_encoder = nn.Identity()

        # --- CFN Recurrent Cell ---
        # Instantiate using the stored kwargs dict
        self.cfn_cell = CFNCell(channels=self.field_channels, **self.cfn_cell_kwargs_config)

        # --- Adaptive Halting Unit ---
        # Instantiate using the stored kwargs dict, overriding input_channels
        _h_kwargs = self.halting_unit_kwargs_config.copy()
        _h_kwargs['input_channels'] = self.field_channels # Ensure correct channel dim
        self.halting_unit = AdaptiveHaltingUnit(**_h_kwargs)

        # --- Output Decoder ---
        if output_decoder is None:
            self.output_decoder = nn.Conv2d(self.field_channels, self.output_raw_channels, kernel_size=1)
            # Simple initialization
            nn.init.kaiming_normal_(self.output_decoder.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') # or linear
            if self.output_decoder.bias is not None: nn.init.zeros_(self.output_decoder.bias)
        else:
            # Use the provided decoder module
            self.output_decoder = output_decoder

    def _check_input_encoder(self, x_raw: torch.Tensor, H_0_features: torch.Tensor):
        """Optional check for input encoder output shape."""
        B, _, H_in, W_in = x_raw.shape
        B_h, C_h, H_h, W_h = H_0_features.shape
        if not (B_h == B and C_h == self.field_channels and H_h == H_in and W_h == W_in):
            raise ValueError(
                f"Input encoder output shape mismatch. Expected (B={B}, C={self.field_channels}, H={H_in}, W={W_in}), "
                f"got {H_0_features.shape} from input {x_raw.shape}"
            )
    def _check_output_decoder(self, H_n: torch.Tensor, y_n: torch.Tensor):
        """Optional check for output decoder input/output shapes."""
        B, C_h, H_h, W_h = H_n.shape
        B_y, C_y, H_y, W_y = y_n.shape
        if not (B_y == B and C_y == self.output_raw_channels and H_y == H_h and W_y == W_h):
             raise ValueError(
                f"Output decoder shape mismatch. Expected (B={B}, C={self.output_raw_channels}, H={H_h}, W={W_h}), "
                f"got {y_n.shape} from input H_n shape {H_n.shape}"
            )

    def forward(self,
                x_raw: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                task_loss_fn_noreduction: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
               ) -> Dict[str, Any]:
        """
        Forward pass for training and 'expected' inference.

        Args:
            x_raw (torch.Tensor): Raw input tensor (B, Cin_raw, H, W).
            target (Optional[torch.Tensor]): Target tensor (B, Cout_raw, H, W).
            task_loss_fn_noreduction (Optional[Callable]): Loss function that returns
                per-element losses, e.g., F.mse_loss(..., reduction='none').
        Returns:
            Dict[str, Any]: Dictionary containing outputs, states, probabilities, and losses.
        """
        batch_size, _, H_raw, W_raw = x_raw.shape
        device = x_raw.device

        # 1. Initial State H_0
        H_0_features = self.input_encoder(x_raw)
        # self._check_input_encoder(x_raw, H_0_features) # Uncomment for debugging shape issues
        current_H = self.positional_encoder(H_0_features)

        y_list, lambda_list = [], []
        active_pixels_mask = torch.ones(batch_size, 1, H_raw, W_raw, device=device, dtype=torch.bool)
        # P_N_sum_check = torch.zeros_like(active_pixels_mask, dtype=torch.float32) # For verifying P_N sums to 1

        # 2. Iterative Refinement Loop
        for n_s in range(self.max_steps): # n_s = number of CFNCell applications (0 to max_steps-1)
            y_n = self.output_decoder(current_H)
            # self._check_output_decoder(current_H, y_n) # Uncomment for debugging shape issues
            y_list.append(y_n)

            lambda_n = self.halting_unit(current_H) # (B, 1, H, W)
            lambda_list.append(lambda_n)

            if n_s < self.max_steps - 1: # Don't update state on the very last loop iteration
                current_H = self.cfn_cell(current_H)

        # 3. Calculate Halting Distribution P_N(n_s | x)
        P_N_list = []
        # cumulative_prod_1_minus_lambda is Prod_{j=0}^{n_s-1} (1 - lambda_j(H_j))
        cumulative_remain_prob = torch.ones_like(lambda_list[0]) # (B, 1, H, W), probability of NOT having halted before step n_s

        for n_s in range(self.max_steps):
            current_lambda_from_H_ns = lambda_list[n_s] # Halting prob calculated based on H_{n_s}

            if n_s == self.max_steps - 1: # Must halt at the last step
                # Probability of halting at last step is remaining probability mass
                p_N_current_ns = cumulative_remain_prob * active_pixels_mask.float()
            else:
                # Probability of halting at this step n_s
                p_N_current_ns = cumulative_remain_prob * current_lambda_from_H_ns * active_pixels_mask.float()

            P_N_list.append(p_N_current_ns)
            # P_N_sum_check += p_N_current_ns # Accumulate for verification

            if n_s < self.max_steps - 1: # Update remaining probability for next step
                cumulative_remain_prob = cumulative_remain_prob * \
                                         (1.0 - current_lambda_from_H_ns + self.eps_stability)
                cumulative_remain_prob.clamp_(min=0.0) # Ensure numerical stability

        P_N_tensor = torch.stack(P_N_list, dim=0) # (max_steps, B, 1, H, W)

        # Verify P_N sums to 1 (approximately) per pixel - can be commented out after testing
        # Check commented out for brevity, was checked before
        # p_n_sum_actual = P_N_tensor.sum(dim=0)
        # if not torch.allclose(p_n_sum_actual, active_pixels_mask.float(), atol=1e-3):
        #     print(f"Warning: P_N sum check min/max: {p_n_sum_actual.min()}, {p_n_sum_actual.max()}. Max diff: {(p_n_sum_actual - active_pixels_mask.float()).abs().max()}")


        # 4. Expected Final Output (for 'expected' inference and training)
        y_list_tensor = torch.stack(y_list, dim=0) # (max_steps, B, Cout_raw, H, W)
        # Weight each step's output y_n by probability P_N[n] and sum
        final_output = (P_N_tensor * y_list_tensor).sum(dim=0) # Shape: (B, Cout_raw, H, W)

        # 5. Ponder Cost & Expected Steps Calculation
        # Cost for PonderNet L_P is generally (number of steps executed).
        # If step n_s is chosen (0 to max_steps-1), the cost is (n_s + 1) PonderNet steps.
        ponder_costs_at_each_n_s = torch.arange(1, self.max_steps + 1, device=device, dtype=P_N_tensor.dtype)
        ponder_costs_at_each_n_s = ponder_costs_at_each_n_s.view(self.max_steps, 1, 1, 1, 1) # Make broadcastable

        expected_ponder_cost_map = (P_N_tensor * ponder_costs_at_each_n_s).sum(dim=0) # (B, 1, H, W)
        L_ponder_scalar_avg = expected_ponder_cost_map.mean() # Scalar average over Batch, Height, Width

        # Expected number of actual CFN cell applications
        num_cell_apps_at_each_n_s = torch.arange(0, self.max_steps, device=device, dtype=P_N_tensor.dtype)
        num_cell_apps_at_each_n_s = num_cell_apps_at_each_n_s.view(self.max_steps, 1, 1, 1, 1) # Make broadcastable
        expected_cell_apps_map = (P_N_tensor * num_cell_apps_at_each_n_s).sum(dim=0) # (B, 1, H, W)
        expected_cell_apps_scalar_avg = expected_cell_apps_map.mean() # Scalar average

        # --- Prepare Results Dictionary ---
        results: Dict[str, Any] = {
            'final_output': final_output,
            'all_step_outputs': y_list_tensor,
            'halting_distribution_p_n': P_N_tensor,
            'all_lambdas': torch.stack(lambda_list, dim=0),
            'expected_ponder_cost_map': expected_ponder_cost_map, # Per pixel expected steps (1 to max_steps)
            'expected_ponder_cost_scalar': L_ponder_scalar_avg, # Average ponder cost (L_P / beta_P)
            'expected_cell_applications_map': expected_cell_apps_map, # Per pixel expected cell calls (0 to max_steps-1)
            'expected_cell_applications_scalar': expected_cell_apps_scalar_avg, # Average cell calls
        }

        # 6. Calculate Losses (only if training and target/loss_fn provided)
        if self.training and target is not None and task_loss_fn_noreduction is not None:
            task_loss_sum = torch.tensor(0.0, device=device)
            for n_s in range(self.max_steps):
                # Get per-element loss for this step's output
                loss_values_n = task_loss_fn_noreduction(y_list[n_s], target) # e.g., (B, Cout_raw, H, W)

                # Weight the per-element loss by the probability of halting at this step
                # P_N_tensor[n_s] is (B, 1, H, W), will broadcast over Cout_raw if loss_values_n has it
                weighted_loss_n = P_N_tensor[n_s] * loss_values_n
                task_loss_sum += weighted_loss_n.sum() # Sum over all elements (pixels, channels)

            # Average the total weighted loss over the batch size
            L_task_scalar_avg = task_loss_sum / batch_size

            # Total loss combines weighted task loss and regularization ponder cost
            # L_ponder_scalar_avg already calculated is the Expected N value for L_P = E[N]
            total_loss = L_task_scalar_avg + self.ponder_reg_beta * L_ponder_scalar_avg

            results['loss_total'] = total_loss
            results['loss_task'] = L_task_scalar_avg
            # L_P in PonderNet paper is the regularization term itself (beta * E[N])
            # Store E[N] as the ponder cost scalar average
            results['loss_ponder_regularization'] = L_ponder_scalar_avg # L_P / beta_P

        return results

    # --- PREDICT SAMPLED METHOD --- (unchanged from previous complete version)
    @torch.no_grad()
    def predict_sampled(self, x_raw: torch.Tensor, sample_per_pixel: bool = False) -> torch.Tensor:
        """
        Inference using sampled number of steps (batch-wise sampling).
        """
        # Ensure model is in evaluation mode
        # assert not self.training, "predict_sampled should only be called in eval mode" # Good check
        if self.training:
             print("Warning: predict_sampled called during training mode. Switching to eval temporarily.")
             self.eval() # Switch to eval mode just for this prediction

        batch_size, _, H_raw, W_raw = x_raw.shape
        device = x_raw.device

        # 1. Initial State H_0
        H_0_features = self.input_encoder(x_raw)
        initial_H_for_all = self.positional_encoder(H_0_features)

        # --- First pass: Collect lambdas to calculate P_N ---
        current_H_for_pn = initial_H_for_all.clone()
        lambda_list_for_pn = []
        for _ in range(self.max_steps):
            lambda_n = self.halting_unit(current_H_for_pn)
            lambda_list_for_pn.append(lambda_n)
            if _ < self.max_steps - 1:
                current_H_for_pn = self.cfn_cell(current_H_for_pn)

        # --- Calculate Halting Distribution P_N ---
        P_N_list_for_pn = []
        cumulative_remain_prob_for_pn = torch.ones_like(lambda_list_for_pn[0])
        for n_s in range(self.max_steps):
            current_lambda_ns = lambda_list_for_pn[n_s]
            if n_s == self.max_steps - 1:
                 p_N_ns = cumulative_remain_prob_for_pn
            else:
                 p_N_ns = cumulative_remain_prob_for_pn * current_lambda_ns
            P_N_list_for_pn.append(p_N_ns)
            if n_s < self.max_steps - 1:
                cumulative_remain_prob_for_pn = cumulative_remain_prob_for_pn * (1.0 - current_lambda_ns + self.eps_stability)
                cumulative_remain_prob_for_pn.clamp_(min=0.0)

        P_N_tensor_for_pn = torch.stack(P_N_list_for_pn, dim=0) # (max_steps, B, 1, H, W)

        # --- Sample N_s (number of cell applications) batch-wise ---
        # Average P_N over spatial dimensions
        P_N_tensor_avg_spatial = P_N_tensor_for_pn.mean(dim=(3,4)).squeeze(-1) # (max_steps, B)
        P_N_tensor_avg_spatial = P_N_tensor_avg_spatial.permute(1,0) # (B, max_steps) for multinomial sampling

        # Normalize probabilities row-wise to ensure they sum to 1 for sampling
        # Use clamp to avoid potential division by zero if all P_N are zero (unlikely but safe)
        p_norm = P_N_tensor_avg_spatial.sum(dim=1, keepdim=True).clamp(min=1e-9)
        P_N_tensor_normalized = P_N_tensor_avg_spatial / p_norm

        # Sample one n_s index (0 to max_steps-1) per batch item
        sampled_n_s_indices = torch.multinomial(P_N_tensor_normalized, num_samples=1).squeeze(-1) # (B,)

        # --- Second pass: Run the model up to the max sampled steps ---
        max_sampled_n_s = sampled_n_s_indices.max().item()

        final_y_sampled = torch.zeros(batch_size, self.output_raw_channels, H_raw, W_raw, device=device)
        current_H_for_eval = initial_H_for_all # Use the already computed H0

        # Iterate from 0 cell applications up to the maximum needed
        for n_s_iter in range(max_sampled_n_s + 1):
            # Find which batch items stop at this number of cell applications
            mask_to_select = (sampled_n_s_indices == n_s_iter) # Shape: (B,)
            if mask_to_select.any():
                # Decode the state H_{n_s_iter} for the items stopping here
                y_final_for_masked = self.output_decoder(current_H_for_eval[mask_to_select])
                # Store the result for these batch items
                final_y_sampled[mask_to_select] = y_final_for_masked

            # Update the state for the next iteration, if it's not the last one needed
            if n_s_iter < max_sampled_n_s:
                current_H_for_eval = self.cfn_cell(current_H_for_eval)

        return final_y_sampled

    # --- EXTRA REPR METHOD --- (unchanged from previous complete version)
    def extra_repr(self) -> str:
        # Representation string
        s = (f"field_channels={self.field_channels}, grid_shape={self.grid_shape}, "
             f"max_steps={self.max_steps}, ponder_reg_beta={self.ponder_reg_beta}\n"
             f"use_pos_enc={self.use_positional_encoding}, inference_mode='{self.inference_mode}'\n"
             f"Input Encoder: {self.input_encoder}\n"
             f"Positional Encoder: {self.positional_encoder}\n"
             f"CFN Cell: {self.cfn_cell}\n"
             f"Halting Unit: {self.halting_unit}\n"
             f"Output Decoder: {self.output_decoder}")
        # Add details about kwargs used for cell and halting unit if desired
        # s += f"\nCFN Cell Kwargs: {self.cfn_cell_kwargs_config}"
        # s += f"\nHalting Unit Kwargs: {self.halting_unit_kwargs_config}"
        return s

# --- __main__ block for testing --- (unchanged from previous complete version)
if __name__ == '__main__':
    print("--- VanillaCFN Example Test (Corrected __init__) ---")

    B, Cin_raw, Cout_raw, H_grid, W_grid = 2, 3, 10, 8, 8 # Smaller grid for faster PE
    FIELD_CHANNELS = 16 # Divisible by 4 for PE
    MAX_STEPS = 4
    PONDER_BETA = 0.01

    # --- Define kwargs dictionaries BEFORE passing ---
    cell_kws = {'kernel_size': 3, 'normalization': 'layer', 'gate_bias_init': 1.0}
    halt_kws = {'halting_bias_init': -2.0, 'kernel_size': 1}
    pos_kws = {'max_grid_shape': (H_grid, W_grid), 'temperature': 1000.0, 'alpha': None} # using alpha=None for fixed PE scale=1

    # Custom Input/Output Encoders (optional)
    custom_input_enc = nn.Sequential(
        nn.Conv2d(Cin_raw, FIELD_CHANNELS // 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(FIELD_CHANNELS // 2, FIELD_CHANNELS, kernel_size=1)
    )
    custom_output_dec = nn.Conv2d(FIELD_CHANNELS, Cout_raw, kernel_size=1)

    # Instantiate model with corrected argument order
    cfn_model = VanillaCFN(
        input_raw_channels=Cin_raw,
        output_raw_channels=Cout_raw,
        field_channels=FIELD_CHANNELS,
        cfn_cell_kwargs=cell_kws,          # Moved earlier
        halting_unit_kwargs=halt_kws,      # Moved earlier
        grid_shape=(H_grid, W_grid),       # Moved later
        max_steps=MAX_STEPS,
        ponder_reg_beta=PONDER_BETA,
        use_positional_encoding=True,      # Uses grid_shape
        pos_enc_kwargs=pos_kws,            # Passed to AddSinePositionalEncoding
        input_encoder=custom_input_enc,    # Using custom ones here
        output_decoder=custom_output_dec,
        inference_mode='expected'
    )

    dummy_x = torch.randn(B, Cin_raw, H_grid, W_grid)
    dummy_target = torch.randn(B, Cout_raw, H_grid, W_grid)

    def mse_loss_no_reduction(pred, target):
        return F.mse_loss(pred, target, reduction='none')

    print("\n--- Training Mode Test ---")
    cfn_model.train()
    results_train = cfn_model(dummy_x, target=dummy_target, task_loss_fn_noreduction=mse_loss_no_reduction)

    for key, val in results_train.items():
        if isinstance(val, torch.Tensor):
            # Avoid printing large tensors, just show shape and maybe one value
             print(f"{key} shape: {val.shape}", end="")
             if val.numel() > 0:
                 print(f", example_val: {val.view(-1)[0].item():.4f}")
             else:
                 print("") # Tensor is empty
        else:
            print(f"{key}: {val}")

    if 'loss_total' in results_train:
        results_train['loss_total'].backward()
        print("Backward pass successful.")
        cfn_model.zero_grad()
    else:
        print("Loss not computed (check target/loss_fn in training).")

    p_n_sum = results_train['halting_distribution_p_n'].sum(dim=0)
    print(f"P_N sum min/max: {p_n_sum.min().item():.4f} / {p_n_sum.max().item():.4f}")
    assert torch.allclose(p_n_sum, torch.ones_like(p_n_sum), atol=1e-2), \
        f"P_N did not sum to 1. Max deviation: {(p_n_sum - 1.0).abs().max()}"

    print("\n--- Inference Mode Test ('expected') ---")
    cfn_model.eval()
    with torch.no_grad(): # Ensure no gradients for eval
        results_eval_exp = cfn_model(dummy_x)
    print(f"Final output shape (expected eval): {results_eval_exp['final_output'].shape}")
    print(f"Expected Ponder Cost (eval): {results_eval_exp['expected_ponder_cost_scalar']:.4f}")

    print("\n--- Inference Mode Test ('predict_sampled' batchwise) ---")
    cfn_model.eval() # Ensure model is in eval mode
    with torch.no_grad(): # Ensure no gradients for sampling prediction
        sampled_output = cfn_model.predict_sampled(dummy_x)
    print(f"Sampled output shape: {sampled_output.shape}")
    assert sampled_output.shape == (B, Cout_raw, H_grid, W_grid)

    print("\nVanillaCFN model init test passed with corrected signature.")
