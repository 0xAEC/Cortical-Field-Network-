# project-root/cfn/vanilla/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable, Tuple, Union

from .cell import CFNCell
from .halting import AdaptiveHaltingUnit
try:
    from ..utils.pos_enc import AddSinePositionalEncoding
except ImportError:
    class AddSinePositionalEncoding(nn.Module):
        def __init__(self, embedding_dim: int, max_grid_shape: Tuple[int, int] = (32,32), **kwargs):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.max_grid_shape = max_grid_shape
            print(f"Warning: cfn.utils.pos_enc.AddSinePositionalEncoding not found or failed to import. Using dummy Identity for PE. Expected embedding_dim={embedding_dim}, max_grid_shape={max_grid_shape}")
        def forward(self, x): return x
        def extra_repr(self) -> str: return "Dummy Identity for Positional Encoding"


class VanillaCFN(nn.Module):
    """
    Vanilla Cortical Field Network (CFN) with PonderNet-style adaptive computation.
    Implements key ideas from CFN 2.0 paper and PonderNet (Banino et al., 2021).

    Args:
        input_raw_channels (int): Channels in the raw input tensor x.
        output_raw_channels (int): Channels in the final task output.
        field_channels (int): Channels C in the internal field state H.
        grid_shape (Tuple[int, int]): Expected (Height, Width) of the input grid.
                                       Required if use_positional_encoding is True.
        cfn_cell_kwargs (dict): Keyword arguments for CFNCell.
        halting_unit_kwargs (dict): Keyword arguments for AdaptiveHaltingUnit.
        max_steps (int): Maximum recurrent steps (N_max in PonderNet).
        ponder_reg_beta (float): Weight for the ponder cost (L_P regularization).
        use_positional_encoding (bool): If True, add positional encodings to H_0.
        pos_enc_kwargs (Optional[dict]): Kwargs for AddSinePositionalEncoding.
                                         'embedding_dim' is field_channels,
                                         'max_grid_shape' is grid_shape.
        input_encoder (Optional[nn.Module]): Encodes raw input to H_0. Default: Conv2D(1x1).
        output_decoder (Optional[nn.Module]): Decodes H_t to task output. Default: Conv2D(1x1).
        eps_stability (float): Small epsilon for numerical stability (e.g., in 1 - lambda).
        inference_mode (Literal['expected', 'sample']): Strategy for eval.
                         'expected': Use weighted average of step outputs.
                         'sample': Sample N_s steps and run for N_s. (Not fully implemented for per-pixel)
    """
    def __init__(
        self,
        input_raw_channels: int,
        output_raw_channels: int,
        field_channels: int,
        grid_shape: Optional[Tuple[int, int]] = None, # H, W
        cfn_cell_kwargs: Dict,
        halting_unit_kwargs: Dict,
        max_steps: int = 10,
        ponder_reg_beta: float = 0.01,
        use_positional_encoding: bool = True,
        pos_enc_kwargs: Optional[Dict] = None,
        input_encoder: Optional[nn.Module] = None,
        output_decoder: Optional[nn.Module] = None,
        eps_stability: float = 1e-8,
        inference_mode: str = 'expected' # 'expected' or 'sample_batchwise'
    ):
        super().__init__()

        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if use_positional_encoding and grid_shape is None:
            raise ValueError("grid_shape must be provided if use_positional_encoding is True.")

        self.input_raw_channels = input_raw_channels
        self.output_raw_channels = output_raw_channels
        self.field_channels = field_channels
        self.grid_shape = grid_shape
        self.max_steps = max_steps
        self.ponder_reg_beta = ponder_reg_beta
        self.use_positional_encoding = use_positional_encoding
        self.eps_stability = eps_stability
        self.inference_mode = inference_mode
        
        # --- Input Encoder ---
        if input_encoder is None:
            self.input_encoder = nn.Conv2d(input_raw_channels, field_channels, kernel_size=1)
            # Simple initialization for default encoder
            nn.init.kaiming_normal_(self.input_encoder.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if self.input_encoder.bias is not None: nn.init.zeros_(self.input_encoder.bias)
        else:
            self.input_encoder = input_encoder
            # Consider adding checks: output should be (B, field_channels, H, W)

        # --- Positional Encoding ---
        if self.use_positional_encoding:
            _pe_kwargs = pos_enc_kwargs if pos_enc_kwargs is not None else {}
            self.positional_encoder = AddSinePositionalEncoding(
                embedding_dim=field_channels,
                max_grid_shape=grid_shape,
                **_pe_kwargs
            )
        else:
            self.positional_encoder = nn.Identity()

        # --- CFN Recurrent Cell ---
        self.cfn_cell = CFNCell(channels=field_channels, **cfn_cell_kwargs)

        # --- Adaptive Halting Unit ---
        _h_kwargs = halting_unit_kwargs.copy()
        _h_kwargs['input_channels'] = field_channels
        self.halting_unit = AdaptiveHaltingUnit(**_h_kwargs)

        # --- Output Decoder ---
        if output_decoder is None:
            self.output_decoder = nn.Conv2d(field_channels, output_raw_channels, kernel_size=1)
            # Simple initialization for default decoder
            nn.init.kaiming_normal_(self.output_decoder.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') # or linear
            if self.output_decoder.bias is not None: nn.init.zeros_(self.output_decoder.bias)

        else:
            self.output_decoder = output_decoder
            # Consider adding checks: input (B, field_channels, H, W), output (B, output_raw_channels, H, W)

    def _check_input_encoder(self, x_raw: torch.Tensor, H_0_features: torch.Tensor):
        B, _, H_in, W_in = x_raw.shape
        B_h, C_h, H_h, W_h = H_0_features.shape
        if not (B_h == B and C_h == self.field_channels and H_h == H_in and W_h == W_in):
            raise ValueError(
                f"Input encoder output shape mismatch. Expected (B, {self.field_channels}, H, W), "
                f"got {H_0_features.shape} from input {x_raw.shape}"
            )
    def _check_output_decoder(self, H_n: torch.Tensor, y_n: torch.Tensor):
        B, _, H_h, W_h = H_n.shape
        B_y, C_y, H_y, W_y = y_n.shape
        if not (B_y == B and C_y == self.output_raw_channels and H_y == H_h and W_y == W_h):
             raise ValueError(
                f"Output decoder shape mismatch. Expected (B, {self.output_raw_channels}, H, W), "
                f"got {y_n.shape} from input {H_n.shape}"
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
        """
        batch_size, _, H_raw, W_raw = x_raw.shape
        device = x_raw.device

        # 1. Initial State H_0
        H_0_features = self.input_encoder(x_raw)
        # self._check_input_encoder(x_raw, H_0_features) # Optional runtime check
        current_H = self.positional_encoder(H_0_features)

        y_list, lambda_list = [], []
        active_pixels_mask = torch.ones(batch_size, 1, H_raw, W_raw, device=device, dtype=torch.bool)
        P_N_sum_check = torch.zeros_like(active_pixels_mask, dtype=torch.float32) # For verifying P_N sums to 1

        for n_s in range(self.max_steps): # n_s = number of CFNCell applications (0 to max_steps-1)
            y_n = self.output_decoder(current_H)
            # self._check_output_decoder(current_H, y_n) # Optional runtime check
            y_list.append(y_n)

            lambda_n = self.halting_unit(current_H) # (B, 1, H, W)
            lambda_list.append(lambda_n)

            if n_s < self.max_steps - 1:
                current_H = self.cfn_cell(current_H)

        # 3. Halting Distribution P_N(n_s | x)
        # P_N_tensor[n_s] is the prob of stopping after n_s CFNCell applications (choosing y_{n_s})
        P_N_list = []
        # cumulative_prod_1_minus_lambda is Prod_{j=0}^{n_s-1} (1 - lambda_j(H_j))
        cumulative_remain_prob = torch.ones_like(lambda_list[0]) # (B, 1, H, W)

        for n_s in range(self.max_steps):
            current_lambda_from_H_ns = lambda_list[n_s] # Halting prob given H_{n_s}

            if n_s == self.max_steps - 1: # Must halt at the last step
                p_N_current_ns = cumulative_remain_prob * active_pixels_mask.float() # Multiply by mask to ensure only active pixels contribute
            else:
                p_N_current_ns = cumulative_remain_prob * current_lambda_from_H_ns * active_pixels_mask.float()
            
            P_N_list.append(p_N_current_ns)
            P_N_sum_check += p_N_current_ns # For verification

            if n_s < self.max_steps - 1:
                cumulative_remain_prob = cumulative_remain_prob * \
                                         (1.0 - current_lambda_from_H_ns + self.eps_stability)
                # Ensure it doesn't go negative due to eps_stability if lambda is 1.
                cumulative_remain_prob.clamp_(min=0.0)
        
        P_N_tensor = torch.stack(P_N_list, dim=0) # (max_steps, B, 1, H, W)
        
        # Verify P_N sums to 1 (approximately) per pixel - can be commented out after testing
        if not torch.allclose(P_N_sum_check, active_pixels_mask.float(), atol=1e-3): # Allow some tolerance
             print(f"Warning: P_N sum check min/max: {P_N_sum_check.min()}, {P_N_sum_check.max()}. Max diff: {(P_N_sum_check - active_pixels_mask.float()).abs().max()}")


        # 4. Expected Final Output (for 'expected' inference and training)
        y_list_tensor = torch.stack(y_list, dim=0) # (max_steps, B, Cout_raw, H, W)
        final_output = (P_N_tensor * y_list_tensor).sum(dim=0)

        # 5. Ponder Cost & Expected Steps
        # Cost for PonderNet L_P is (num_cfn_cell_ops + 1)
        ponder_costs_at_each_n_s = torch.arange(1, self.max_steps + 1, device=device, dtype=P_N_tensor.dtype)
        ponder_costs_at_each_n_s = ponder_costs_at_each_n_s.view(self.max_steps, 1, 1, 1, 1)
        
        expected_ponder_cost_map = (P_N_tensor * ponder_costs_at_each_n_s).sum(dim=0) # (B, 1, H, W)
        L_ponder_scalar_avg = expected_ponder_cost_map.mean() # Average over B, H, W

        # Expected number of CFN cell applications
        num_cell_apps_at_each_n_s = torch.arange(0, self.max_steps, device=device, dtype=P_N_tensor.dtype)
        num_cell_apps_at_each_n_s = num_cell_apps_at_each_n_s.view(self.max_steps, 1, 1, 1, 1)
        expected_cell_apps_map = (P_N_tensor * num_cell_apps_at_each_n_s).sum(dim=0)
        expected_cell_apps_scalar_avg = expected_cell_apps_map.mean()

        results: Dict[str, Any] = {
            'final_output': final_output,
            'all_step_outputs': y_list_tensor,
            'halting_distribution_p_n': P_N_tensor,
            'all_lambdas': torch.stack(lambda_list, dim=0),
            'expected_ponder_cost_map': expected_ponder_cost_map,
            'expected_ponder_cost_scalar': L_ponder_scalar_avg,
            'expected_cell_applications_map': expected_cell_apps_map,
            'expected_cell_applications_scalar': expected_cell_apps_scalar_avg,
        }

        if self.training and target is not None and task_loss_fn_noreduction is not None:
            task_loss_sum = torch.tensor(0.0, device=device)
            for n_s in range(self.max_steps):
                # task_loss_fn_noreduction should return per-element losses
                # e.g., for MSE shape (B, Cout_raw, H, W)
                loss_values_n = task_loss_fn_noreduction(y_list[n_s], target) # (B, Cout_raw, H, W) or (B, H, W) etc.
                
                # Weight per-element losses by P_N_tensor[n_s]
                # P_N_tensor[n_s] is (B, 1, H, W), broadcasts over Cout_raw
                weighted_loss_n = P_N_tensor[n_s] * loss_values_n # Element-wise
                task_loss_sum += weighted_loss_n.sum() # Sum all weighted element losses

            # Average over total number of elements (Batch * Cout_raw * H * W) and steps implicitly by summing weighted P_N
            # A more direct average per batch item would be .sum() / (target.numel() / batch_size)
            L_task_scalar_avg = task_loss_sum / batch_size # Average over batch
            # Alternative: L_task_scalar_avg = task_loss_sum / target.numel() if total_loss is truly sum of all pixel losses

            total_loss = L_task_scalar_avg + self.ponder_reg_beta * L_ponder_scalar_avg

            results['loss_total'] = total_loss
            results['loss_task'] = L_task_scalar_avg
            results['loss_ponder_regularization'] = L_ponder_scalar_avg # This is L_P from paper
        
        # For inference mode 'sample_batchwise'
        # This would involve sampling N_s per batch item from P_N_tensor distribution
        # then re-running a loop up to N_s for each item. More complex.
        # Current setup mainly supports 'expected' inference or sampling outside.
            
        return results
    
    @torch.no_grad()
    def predict_sampled(self, x_raw: torch.Tensor, sample_per_pixel: bool = False) -> torch.Tensor:
        """
        Inference using sampled number of steps. This is more computationally intensive
        as it may require re-running parts of the loop or careful state management if
        loops are dynamic per batch item or pixel.
        
        A simpler batch-wise sampling is shown here for demonstration.
        Per-pixel sampling is significantly more complex to implement efficiently.
        """
        self.eval()
        batch_size, _, H_raw, W_raw = x_raw.shape
        device = x_raw.device

        # Get H_0
        H_0_features = self.input_encoder(x_raw)
        initial_H_for_all = self.positional_encoder(H_0_features)
        
        # --- First pass: get all lambdas and P_N to sample N_s ---
        # This part is similar to the training forward pass to get P_N
        current_H_for_pn = initial_H_for_all.clone()
        lambda_list_for_pn = []
        for _ in range(self.max_steps):
            lambda_list_for_pn.append(self.halting_unit(current_H_for_pn))
            if _ < self.max_steps - 1:
                current_H_for_pn = self.cfn_cell(current_H_for_pn)
        
        P_N_list_for_pn = []
        cumulative_remain_prob_for_pn = torch.ones_like(lambda_list_for_pn[0])
        for n_s in range(self.max_steps):
            current_lambda_ns = lambda_list_for_pn[n_s]
            if n_s == self.max_steps - 1: p_N_ns = cumulative_remain_prob_for_pn
            else: p_N_ns = cumulative_remain_prob_for_pn * current_lambda_ns
            P_N_list_for_pn.append(p_N_ns)
            if n_s < self.max_steps - 1:
                cumulative_remain_prob_for_pn = cumulative_remain_prob_for_pn * (1.0 - current_lambda_ns + self.eps_stability)
                cumulative_remain_prob_for_pn.clamp_(min=0.0)
        
        P_N_tensor_for_pn = torch.stack(P_N_list_for_pn, dim=0) # (max_steps, B, 1, H, W)

        # --- Sample N_s (number of cell applications) ---
        # For simplicity, sample one N_s per batch item by averaging P_N spatially
        # P_N_tensor_for_pn_avg_spatial = P_N_tensor_for_pn.mean(dim=(2,3,4)) # (max_steps, B)
        # This should be mean over H, W. Channel dim is 1. So mean(dim=(2,3)) -> (max_steps, B, 1) -> squeeze
        P_N_tensor_for_pn_avg_spatial = P_N_tensor_for_pn.mean(dim=(3,4)).squeeze(-1) # (max_steps, B)
        P_N_tensor_for_pn_avg_spatial = P_N_tensor_for_pn_avg_spatial.permute(1,0) # (B, max_steps) for Multinomial
        
        # Ensure probabilities sum to 1 for Multinomial.
        # Normalize due to potential small float inaccuracies, especially from eps_stability.
        P_N_tensor_for_pn_avg_spatial = F.normalize(P_N_tensor_for_pn_avg_spatial, p=1, dim=1)

        sampled_n_s_indices = torch.multinomial(P_N_tensor_for_pn_avg_spatial, num_samples=1).squeeze(-1) # (B,)
                                                                              # these are indices for max_steps, so 0 to max_steps-1

        # --- Second pass: run loop up to sampled_n_s_indices for each batch item ---
        # This is tricky with batched operations if N_s varies per item.
        # Easiest: find max(sampled_n_s_indices) and run everyone up to that, then select.
        max_sampled_n_s = sampled_n_s_indices.max().item()
        
        final_y_sampled = torch.zeros(batch_size, self.output_raw_channels, H_raw, W_raw, device=device)
        current_H_for_eval = initial_H_for_all.clone()

        for n_s_iter in range(max_sampled_n_s + 1): # Iterate up to the max N_s sampled
            # For batch items where sampled_n_s_indices == n_s_iter, this is their final H state.
            mask_to_select = (sampled_n_s_indices == n_s_iter) # (B,)
            if mask_to_select.any():
                # Select relevant H states, decode, and store
                y_final_for_masked = self.output_decoder(current_H_for_eval[mask_to_select])
                final_y_sampled[mask_to_select] = y_final_for_masked
            
            if n_s_iter < max_sampled_n_s: # Don't update if it's the last required iteration
                current_H_for_eval = self.cfn_cell(current_H_for_eval)
                
        return final_y_sampled


    def extra_repr(self) -> str:
        s = (f"field_channels={self.field_channels}, grid_shape={self.grid_shape}, "
             f"max_steps={self.max_steps}, ponder_reg_beta={self.ponder_reg_beta}\n"
             f"use_pos_enc={self.use_positional_encoding}, inference_mode='{self.inference_mode}'\n"
             f"Input Encoder: {self.input_encoder}\n"
             f"Positional Encoder: {self.positional_encoder}\n"
             f"CFN Cell: {self.cfn_cell}\n"
             f"Halting Unit: {self.halting_unit}\n"
             f"Output Decoder: {self.output_decoder}")
        return s

if __name__ == '__main__':
    print("--- VanillaCFN Example Test (Refined) ---")

    B, Cin_raw, Cout_raw, H_grid, W_grid = 2, 3, 10, 8, 8 # Smaller grid for faster PE
    FIELD_CHANNELS = 16 # Divisible by 4 for PE
    MAX_STEPS = 4
    PONDER_BETA = 0.01

    cell_kws = {'kernel_size': 3, 'normalization': 'layer', 'gate_bias_init': 1.0}
    halt_kws = {'halting_bias_init': -2.0, 'kernel_size': 1}
    # Pos enc kwargs: alpha=None for fixed scale, or provide a float for learnable
    pos_kws = {'max_grid_shape': (H_grid, W_grid), 'temperature': 1000.0, 'alpha': None}

    # Custom Input Encoder (example: deeper than 1x1)
    custom_input_enc = nn.Sequential(
        nn.Conv2d(Cin_raw, FIELD_CHANNELS // 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(FIELD_CHANNELS // 2, FIELD_CHANNELS, kernel_size=1)
    )
    # Custom Output Decoder (example)
    custom_output_dec = nn.Conv2d(FIELD_CHANNELS, Cout_raw, kernel_size=1)

    cfn_model = VanillaCFN(
        input_raw_channels=Cin_raw, output_raw_channels=Cout_raw,
        field_channels=FIELD_CHANNELS, grid_shape=(H_grid, W_grid),
        cfn_cell_kwargs=cell_kws, halting_unit_kwargs=halt_kws,
        max_steps=MAX_STEPS, ponder_reg_beta=PONDER_BETA,
        use_positional_encoding=True, pos_enc_kwargs=pos_kws,
        input_encoder=custom_input_enc, output_decoder=custom_output_dec,
        inference_mode='expected'
    )
    # print("\nVanillaCFN Model Structure:")
    # print(cfn_model) # Very verbose

    dummy_x = torch.randn(B, Cin_raw, H_grid, W_grid)
    dummy_target = torch.randn(B, Cout_raw, H_grid, W_grid)

    def mse_loss_no_reduction(pred, target):
        return F.mse_loss(pred, target, reduction='none')

    print("\n--- Training Mode Test ---")
    cfn_model.train()
    results_train = cfn_model(dummy_x, target=dummy_target, task_loss_fn_noreduction=mse_loss_no_reduction)

    for key, val in results_train.items():
        if isinstance(val, torch.Tensor):
            print(f"{key} shape: {val.shape}, example_val: {val.flatten()[0].item():.4f}")
        else:
            print(f"{key}: {val}")
    
    if 'loss_total' in results_train:
        results_train['loss_total'].backward()
        print("Backward pass successful.")
        cfn_model.zero_grad()
    else:
        print("Loss not computed (check target/loss_fn).")

    p_n_sum = results_train['halting_distribution_p_n'].sum(dim=0)
    print(f"P_N sum min/max: {p_n_sum.min().item():.4f} / {p_n_sum.max().item():.4f}")
    # Assert this outside for clarity in test, it might fail with low MAX_STEPS if lambda small
    assert torch.allclose(p_n_sum, torch.ones_like(p_n_sum), atol=1e-2), \
        f"P_N did not sum to 1. Max deviation: {(p_n_sum - 1.0).abs().max()}"


    print("\n--- Inference Mode Test ('expected') ---")
    cfn_model.eval()
    results_eval_exp = cfn_model(dummy_x)
    print(f"Final output shape (expected eval): {results_eval_exp['final_output'].shape}")
    print(f"Expected Ponder Cost (eval): {results_eval_exp['expected_ponder_cost_scalar']:.4f}")
    
    print("\n--- Inference Mode Test ('predict_sampled' batchwise) ---")
    cfn_model.eval() # Ensure model is in eval mode
    sampled_output = cfn_model.predict_sampled(dummy_x)
    print(f"Sampled output shape: {sampled_output.shape}")
    assert sampled_output.shape == (B, Cout_raw, H_grid, W_grid)


    print("\nVanillaCFN refined example tests logic appears OK.")
