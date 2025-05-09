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
# If run from 0xaec-cortical-field-network- (one level up from project-root), these are correct
from cfn.vanilla.model import VanillaCFN
from data.tasks import get_denoising_mnist_loaders # Assumes MNIST denoising task
# For a generic setup, you might import a more general data factory function

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

    # Model args: Input/Output Encoders (simplified for now, can extend to load custom modules)
    # For this example, we use default 1x1 convs or slightly deeper ones based on task.

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
    if args.cell_normalization.lower() == 'none':
        args.cell_normalization = None
        
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
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
     # === Start Paste ===
     args_dict = vars(args)
     # Convert non-serializable items (like torch.device) to strings
     serializable_args = {}
     for key, value in args_dict.items():
         if isinstance(value, torch.device):
             serializable_args[key] = str(value) # Store 'cuda' or 'cpu' as string
         # Add other type checks here if needed (e.g., for function objects)
         elif callable(value): # Example: skip functions if any sneaked in
             serializable_args[key] = f"<function {value.__name__}>" # Or just skip
         else:
             try:
                 # Attempt to include value, skip if truly not serializable
                 json.dumps(value) # Test serializability
                 serializable_args[key] = value
             except TypeError:
                 serializable_args[key] = f"<non-serializable type: {type(value).__name__}>"

     # Dump the serializable dictionary
     json.dump(serializable_args, f, indent=4)
     # === End Paste ===

    print(f"Using device: {args.device}")
    print("Config:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 30)

    # --- Data ---
    # For MNIST denoising:
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
    # Simple input/output encoders for denoising task
    # Can be made more sophisticated
    input_encoder = nn.Conv2d(input_raw_channels, args.field_channels, kernel_size=3, padding=1)
    output_decoder = nn.Conv2d(args.field_channels, output_raw_channels, kernel_size=3, padding=1)
    
    # Initialize encoders explicitly for this example
    nn.init.kaiming_normal_(input_encoder.weight, nonlinearity='relu')
    if input_encoder.bias is not None: nn.init.zeros_(input_encoder.bias)
    nn.init.kaiming_normal_(output_decoder.weight, nonlinearity='linear') # Final layer often linear
    if output_decoder.bias is not None: nn.init.zeros_(output_decoder.bias)


    model = VanillaCFN(
        input_raw_channels=input_raw_channels,
        output_raw_channels=output_raw_channels,
        field_channels=args.field_channels,
        grid_shape=grid_shape,
        cfn_cell_kwargs=cfn_cell_kwargs,
        halting_unit_kwargs=halting_unit_kwargs,
        max_steps=args.max_steps,
        ponder_reg_beta=args.ponder_reg_beta,
        use_positional_encoding=not args.disable_pos_enc,
        pos_enc_kwargs={'max_grid_shape': grid_shape, 'alpha': 1.0, 'scale_factor_learnable': False}, # Example
        input_encoder=input_encoder,
        output_decoder=output_decoder,
    ).to(args.device)

    print(f"\nModel created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    # print(model) # Can be very verbose

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Consider a learning rate scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 

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
            
            # Forward pass
            outputs = model(x_raw, target=target, task_loss_fn_noreduction=task_loss_fn)
            
            loss = outputs.get('loss_total')
            if loss is None:
                print("Warning: Total loss not found in model outputs during training.")
                continue
            
            loss.backward()
            if args.grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
            optimizer.step()

            total_train_loss += loss.item()
            total_train_task_loss += outputs.get('loss_task', torch.tensor(0.0)).item()
            total_train_ponder_loss += outputs.get('loss_ponder_regularization', torch.tensor(0.0)).item()
            total_train_expected_steps += outputs.get('expected_ponder_cost_scalar', torch.tensor(0.0)).item() # Use ponder cost here

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = total_train_loss / args.log_interval
                avg_task_loss = total_train_task_loss / args.log_interval
                avg_ponder_loss = total_train_ponder_loss / args.log_interval
                avg_expected_steps = total_train_expected_steps / args.log_interval
                
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'TaskL': f'{avg_task_loss:.4f}',
                    'PondL': f'{avg_ponder_loss:.4f}',
                    'ExpSteps': f'{avg_expected_steps:.2f}'
                })
                total_train_loss, total_train_task_loss, total_train_ponder_loss, total_train_expected_steps = 0,0,0,0
        
        # if scheduler: scheduler.step()

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
                
                loss = outputs.get('loss_total', torch.tensor(0.0)) # Handle if not present (e.g. if model.eval behavior changes)
                total_val_loss += loss.item()
                total_val_task_loss += outputs.get('loss_task', torch.tensor(0.0)).item()
                total_val_ponder_loss += outputs.get('loss_ponder_regularization', torch.tensor(0.0)).item()
                total_val_expected_steps += outputs.get('expected_ponder_cost_scalar', torch.tensor(0.0)).item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_task_loss = total_val_task_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_ponder_loss = total_val_ponder_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_expected_steps = total_val_expected_steps / num_val_batches if num_val_batches > 0 else 0
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} Summary: Val_Loss: {avg_val_loss:.4f} | Val_TaskL: {avg_val_task_loss:.4f} | "
              f"Val_PondL: {avg_val_ponder_loss:.4f} | Val_ExpSteps: {avg_val_expected_steps:.2f} | Time: {epoch_duration:.2f}s")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(args.save_dir, f'best_model_epoch{epoch}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model to {model_path} (Val Loss: {best_val_loss:.4f})")
        
        # Save latest model
        latest_model_path = os.path.join(args.save_dir, 'latest_model.pt')
        torch.save(model.state_dict(), latest_model_path)

    print("Training finished.")
    final_model_path = os.path.join(args.save_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    args = get_args()
    train(args)
