# project-root/cfn/train/train_vanilla.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import json
from tqdm import tqdm # For progress bars
import time

# Make sure to adjust relative imports based on your project structure if run from project-root
# This import assumes the script is run from project-root and PYTHONPATH="." is set,
# or if run from parent using python -m project-root.cfn.train.train_vanilla
from cfn.vanilla.model import VanillaCFN
# This import assumes data/tasks.py is a top-level package/module relative to project-root
from data.tasks import get_denoising_mnist_loaders # Assumes MNIST denoising task

# --- Configuration & Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Train Vanilla CFN with PonderNet")

    # Data args
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--noise_factor', type=float, default=0.3, help='Noise factor for denoising task')

    # Model args: VanillaCFN general
    parser.add_argument('--field_channels', type=int, default=32, help='Number of channels in CFN field state H')
    parser.add_argument('--max_steps', type=int, default=8, help='Maximum recurrent steps (N_max for PonderNet)')
    parser.add_argument('--ponder_reg_beta', type=float, default=0.01, help='Ponder cost regularization beta')
    parser.add_argument('--disable_pos_enc', action='store_true', help='Disable positional encoding')

    # Model args: CFNCell specific (passed as dict)
    parser.add_argument('--cell_kernel_size', type=int, default=3)
    parser.add_argument('--cell_padding_mode', type=str, default='replicate', choices=['zeros', 'reflect', 'replicate', 'circular'])
    parser.add_argument('--cell_normalization', type=str, default='layer', choices=['group', 'layer', 'none'], help="Use 'none' for None")
    parser.add_argument('--cell_group_norm_groups', type=int, default=1) # Only if cell_normalization is 'group'
    parser.add_argument('--cell_bias', type=lambda x: (str(x).lower() == 'true'), default=True) # bool
    parser.add_argument('--cell_gate_bias_init', type=float, default=2.0)

    # Model args: HaltingUnit specific (passed as dict)
    parser.add_argument('--halt_kernel_size', type=int, default=1)
    parser.add_argument('--halt_bias_init', type=float, default=-3.0)

    # Training args
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--grad_clip_value', type=float, default=1.0, help='Value for gradient clipping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--log_interval', type=int, default=50, help='How many batches to wait before logging training status')
    parser.add_argument('--save_dir', type=str, default='./results/vanilla_cfn_denoising', help='Directory to save results and checkpoints')
    parser.add_argument('--no_cuda', action='store_true', help='Disables CUDA training')

    args = parser.parse_args()

    # Post-process some args
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    if args.cell_normalization is not None and args.cell_normalization.lower() == 'none':
        args.cell_normalization = None # Ensure None type if 'none' string is passed

    return args

# --- Task-Specific Loss Function ---
def mse_loss_no_reduction(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ MSE loss with reduction='none' for PonderNet per-pixel weighting. """
    return F.mse_loss(pred, target, reduction='none')

# --- Main Training Function ---
def train(args):
    # Setup
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    # --- Corrected code for saving args ---
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        # Intentionally placed block with correct indentation
        args_dict = vars(args)
        serializable_args = {}
        for key, value in args_dict.items():
            if isinstance(value, torch.device):
                serializable_args[key] = str(value)
            elif callable(value):
                serializable_args[key] = f"<function {value.__name__}>"
            else:
                try:
                    json.dumps(value)
                    serializable_args[key] = value
                except TypeError:
                    serializable_args[key] = f"<non-serializable type: {type(value).__name__}>"
        json.dump(serializable_args, f, indent=4)
    # --- End Corrected code ---

    print(f"Using device: {args.device}")
    print("Config:")
    # Print the *serializable* args to console as well for confirmation
    for k, v in serializable_args.items(): # Use serializable_args here
        print(f"  {k}: {v}")
    print("-" * 30)

    # --- Data ---
    input_raw_channels = 1
    output_raw_channels = 1
    grid_shape = (28, 28) # H, W for MNIST
    train_loader, val_loader = get_denoising_mnist_loaders(
        batch_size=args.batch_size,
        noise_factor=args.noise_factor,
        data_root_dir=args.data_root
    )
    task_loss_fn = mse_loss_no_reduction

    # --- Model ---
    cfn_cell_kwargs = {
        'kernel_size': args.cell_kernel_size,
        'padding_mode': args.cell_padding_mode,
        'normalization': args.cell_normalization,
        'group_norm_num_groups': args.cell_group_norm_groups,
        'bias': args.cell_bias,
        'gate_bias_init': args.cell_gate_bias_init
    }
    halting_unit_kwargs = {
        'halting_bias_init': args.halt_bias_init,
        'kernel_size': args.halt_kernel_size
    }
    # Use 3x3 encoders/decoders for this denoising task example
    input_encoder = nn.Conv2d(input_raw_channels, args.field_channels, kernel_size=3, padding=1)
    output_decoder = nn.Conv2d(args.field_channels, output_raw_channels, kernel_size=3, padding=1)

    # Initialize these encoders/decoders
    nn.init.kaiming_normal_(input_encoder.weight, nonlinearity='relu')
    if input_encoder.bias is not None: nn.init.zeros_(input_encoder.bias)
    nn.init.kaiming_normal_(output_decoder.weight, nonlinearity='linear')
    if output_decoder.bias is not None: nn.init.zeros_(output_decoder.bias)

    # Define pos_enc_kwargs but remove 'max_grid_shape' as it's passed via grid_shape arg now
    # Ensure alpha related keys exist even if PE is disabled, or handle it in model __init__ better.
    # Let's define it conditionally:
# train_vanilla.py - CORRECTED CODE block ts is to argh
pos_enc_specific_kwargs = {}
if not args.disable_pos_enc:
     pos_enc_specific_kwargs = {
         # Define the *arguments expected by AddSinePositionalEncoding2D*
         'scale_factor_learnable': False,  # Example: Fixed scaling factor
         'initial_scale_factor': 1.0,    # Example: The value of the fixed factor is 1.0
         'temperature': 10000.0,         # Optionally pass other PE params here too
         'normalize_pe': False,
         'dropout_prob': 0.0
         # Add any other valid AddSinePositionalEncoding2D arguments here if needed
     }

    # Instantiate VanillaCFN model
    model = VanillaCFN(
        input_raw_channels=input_raw_channels,
        output_raw_channels=output_raw_channels,
        field_channels=args.field_channels,
        cfn_cell_kwargs=cfn_cell_kwargs,          # REQUIRED non-default
        halting_unit_kwargs=halting_unit_kwargs,  # REQUIRED non-default
        grid_shape=grid_shape if not args.disable_pos_enc else None, # Pass only if needed
        max_steps=args.max_steps,
        ponder_reg_beta=args.ponder_reg_beta,
        use_positional_encoding=not args.disable_pos_enc,
        pos_enc_kwargs=pos_enc_specific_kwargs,   # Use the filtered dict
        input_encoder=input_encoder,              # Pass custom encoder
        output_decoder=output_decoder,            # Pass custom decoder
    ).to(args.device)

    print(f"\nModel created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Training Loop ---
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_train_task_loss = 0
        total_train_ponder_loss = 0
        total_train_expected_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, (x_raw, target) in enumerate(progress_bar):
            x_raw, target = x_raw.to(args.device), target.to(args.device)

            optimizer.zero_grad()

            outputs = model(x_raw, target=target, task_loss_fn_noreduction=task_loss_fn)

            loss = outputs.get('loss_total')
            if loss is None:
                print("Warning: Total loss not found in model outputs during training.")
                continue # Skip backprop if loss is missing

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"ERROR: NaN loss encountered at epoch {epoch}, batch {batch_idx}. Stopping.")
                # Optionally save state or raise error
                # torch.save(model.state_dict(), os.path.join(args.save_dir, 'nan_error_model.pt'))
                return # Stop training

            loss.backward()
            if args.grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
            optimizer.step()

            # Accumulate stats (ensure items are detached if accumulating gradients is not intended)
            total_train_loss += loss.item()
            total_train_task_loss += outputs.get('loss_task', torch.tensor(0.0)).item()
            total_train_ponder_loss += outputs.get('loss_ponder_regularization', torch.tensor(0.0)).item()
            # Use expected_ponder_cost_scalar which is E[N] where N=num_steps (1..max_steps)
            total_train_expected_steps += outputs.get('expected_ponder_cost_scalar', torch.tensor(0.0)).item()

            if (batch_idx + 1) % args.log_interval == 0:
                # Calculate averages for the interval
                samples_in_interval = args.log_interval * args.batch_size # Adjust if last batch smaller? Roughly ok.
                avg_loss = total_train_loss / args.log_interval
                avg_task_loss = total_train_task_loss / args.log_interval
                avg_ponder_loss = total_train_ponder_loss / args.log_interval
                avg_expected_steps = total_train_expected_steps / args.log_interval

                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'TaskL': f'{avg_task_loss:.4f}',
                    'PondL': f'{avg_ponder_loss:.4f}', # This is L_P / beta_P = E[N]
                    'ExpSteps': f'{avg_expected_steps:.2f}' # E[N] is the expected number of steps
                })
                # Reset interval accumulators
                total_train_loss, total_train_task_loss, total_train_ponder_loss, total_train_expected_steps = 0, 0, 0, 0

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        total_val_task_loss = 0
        total_val_ponder_loss = 0
        total_val_expected_steps = 0
        num_val_batches = 0

        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val  ]")
        with torch.no_grad():
            for x_raw, target in val_progress_bar:
                x_raw, target = x_raw.to(args.device), target.to(args.device)
                outputs = model(x_raw, target=target, task_loss_fn_noreduction=task_loss_fn)

                # Calculate loss for validation (model computes it internally if target provided)
                loss = outputs.get('loss_total')
                if loss is not None:
                     total_val_loss += loss.item()
                     total_val_task_loss += outputs.get('loss_task', torch.tensor(0.0)).item()
                     total_val_ponder_loss += outputs.get('loss_ponder_regularization', torch.tensor(0.0)).item()
                     total_val_expected_steps += outputs.get('expected_ponder_cost_scalar', torch.tensor(0.0)).item()
                else:
                     # If loss calculation relies on self.training, we might need to manually calc val loss
                     # Or just rely on task-specific metrics like MSE/PSNR from final_output
                     print("Warning: Total loss not found during validation.")

                num_val_batches += 1

        # Calculate averages
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_task_loss = total_val_task_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_ponder_loss = total_val_ponder_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_expected_steps = total_val_expected_steps / num_val_batches if num_val_batches > 0 else 0

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} Summary: Val_Loss: {avg_val_loss:.4f} | Val_TaskL: {avg_val_task_loss:.4f} | "
              f"Val_PondL(E[N]): {avg_val_ponder_loss:.4f} | Val_ExpSteps: {avg_val_expected_steps:.2f} | Time: {epoch_duration:.2f}s")

        # Save best model based on validation loss
        if num_val_batches > 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                 model_path = os.path.join(args.save_dir, f'best_model_epoch_{epoch}_loss_{best_val_loss:.4f}.pt')
                 torch.save(model.state_dict(), model_path)
                 print(f"Saved new best model to {model_path} (Val Loss: {best_val_loss:.4f})")
            except Exception as e:
                 print(f"Error saving best model: {e}")


        # Save latest model checkpoint
        try:
            latest_model_path = os.path.join(args.save_dir, 'latest_model.pt')
            torch.save({
                 'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'best_val_loss': best_val_loss,
                 # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                 'args': serializable_args # Save args used
            }, latest_model_path)
        except Exception as e:
             print(f"Error saving latest model checkpoint: {e}")


    print("Training finished.")
    try:
         final_model_path = os.path.join(args.save_dir, 'final_model.pt')
         torch.save(model.state_dict(), final_model_path)
         print(f"Final model saved to {final_model_path}")
    except Exception as e:
         print(f"Error saving final model: {e}")


if __name__ == '__main__':
    args = get_args()
    # Basic check before running train
    if args.field_channels % 4 != 0 and not args.disable_pos_enc:
        print(f"Warning: field_channels ({args.field_channels}) is not divisible by 4. "
              "SinePositionalEncoding2D expects this if enabled.")
    train(args)
