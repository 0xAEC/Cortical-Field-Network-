# project-root/cfn/train/train_vanilla.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import json
from tqdm import tqdm 
import time

# This import assumes the script is run from project-root (CWD) and PYTHONPATH="." is effectively set,this is also crucial for me uhh when running the training script.... 
# making 'cfn' and 'data' top-level packages for the interpreter.
from cfn.vanilla.model import VanillaCFN
from data.tasks import get_denoising_mnist_loaders # Assumes MNIST denoising task

# --- Configuration & Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Train Vanilla CFN with PonderNet")

    # Data args
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for datasets relative to project-root')
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
    parser.add_argument('--cell_group_norm_groups', type=int, default=1)
    parser.add_argument('--cell_bias', type=lambda x: (str(x).lower() == 'true'), default=True)
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
    parser.add_argument('--save_dir', type=str, default='./results/vanilla_cfn_denoising', help='Directory to save results (relative to project-root)')
    parser.add_argument('--no_cuda', action='store_true', help='Disables CUDA training')

    args = parser.parse_args()

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    if args.cell_normalization is not None and args.cell_normalization.lower() == 'none':
        args.cell_normalization = None

    return args

def mse_loss_no_reduction(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction='none')

def train(args):
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Make paths absolute based on the script's presumed parent (project-root)
    # Assumes this train_vanilla.py script is inside project-root/cfn/train/
    script_location_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(script_location_dir, '..', '..')) # Up two levels to project-root

    save_dir = os.path.join(project_root_dir, args.save_dir)
    data_root = os.path.join(project_root_dir, args.data_root) # For data downloaded by the script

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        args_dict = vars(args)
        serializable_args = {}
        for key, value in args_dict.items():
            if isinstance(value, torch.device):
                serializable_args[key] = str(value)
            elif callable(value):
                serializable_args[key] = f"<function {value.__name__}>"
            else:
                try:
                    json.dumps(value) # Test serializability
                    serializable_args[key] = value
                except TypeError:
                    serializable_args[key] = f"<non-serializable type: {type(value).__name__}>"
        json.dump(serializable_args, f, indent=4)

    print(f"Using device: {args.device}")
    print("Effective Config:")
    temp_display_args = serializable_args.copy()
    temp_display_args['save_dir_resolved'] = save_dir
    temp_display_args['data_root_resolved'] = data_root
    for k, v in temp_display_args.items():
        print(f"  {k}: {v}")
    print("-" * 30)

    input_raw_channels = 1
    output_raw_channels = 1
    grid_shape = (28, 28)
    train_loader, val_loader = get_denoising_mnist_loaders(
        batch_size=args.batch_size,
        noise_factor=args.noise_factor,
        data_root_dir=data_root # Use resolved data_root
    )
    task_loss_fn = mse_loss_no_reduction

    cfn_cell_kwargs = {
        'kernel_size': args.cell_kernel_size, 'padding_mode': args.cell_padding_mode,
        'normalization': args.cell_normalization, 'group_norm_num_groups': args.cell_group_norm_groups,
        'bias': args.cell_bias, 'gate_bias_init': args.cell_gate_bias_init
    }
    halting_unit_kwargs = {
        'halting_bias_init': args.halt_bias_init, 'kernel_size': args.halt_kernel_size
    }
    input_encoder = nn.Conv2d(input_raw_channels, args.field_channels, kernel_size=3, padding=1)
    output_decoder = nn.Conv2d(args.field_channels, output_raw_channels, kernel_size=3, padding=1)
    nn.init.kaiming_normal_(input_encoder.weight, nonlinearity='relu')
    if input_encoder.bias is not None: nn.init.zeros_(input_encoder.bias)
    nn.init.kaiming_normal_(output_decoder.weight, nonlinearity='linear')
    if output_decoder.bias is not None: nn.init.zeros_(output_decoder.bias)

    pos_enc_specific_kwargs = {}
    if not args.disable_pos_enc:
        pos_enc_specific_kwargs = {
            'scale_factor_learnable': False,
            'initial_scale_factor': 1.0,
            'temperature': 10000.0,
            'normalize_pe': False,
            'dropout_prob': 0.0
        }

    # INDENTATION OF THIS BLOCK IS CRITICAL
    model = VanillaCFN(
        input_raw_channels=input_raw_channels,
        output_raw_channels=output_raw_channels,
        field_channels=args.field_channels,
        cfn_cell_kwargs=cfn_cell_kwargs,
        halting_unit_kwargs=halting_unit_kwargs,
        grid_shape=grid_shape if not args.disable_pos_enc else None,
        max_steps=args.max_steps,
        ponder_reg_beta=args.ponder_reg_beta,
        use_positional_encoding=not args.disable_pos_enc,
        pos_enc_kwargs=pos_enc_specific_kwargs,
        input_encoder=input_encoder,
        output_decoder=output_decoder,
    ).to(args.device)

    print(f"\nModel created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_train_loss_epoch = 0
        total_train_task_loss_epoch = 0
        total_train_ponder_loss_epoch = 0
        total_train_expected_steps_epoch = 0
        num_train_batches_epoch = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, (x_raw, target) in enumerate(progress_bar):
            x_raw, target = x_raw.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            outputs = model(x_raw, target=target, task_loss_fn_noreduction=task_loss_fn)
            loss = outputs.get('loss_total')

            if loss is None:
                print(f"Warning: Total loss is None at epoch {epoch}, batch {batch_idx}.")
                continue
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: NaN or Inf loss encountered at epoch {epoch}, batch {batch_idx}. Loss: {loss.item()}. Stopping.")
                return

            loss.backward()
            if args.grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
            optimizer.step()

            total_train_loss_epoch += loss.item()
            total_train_task_loss_epoch += outputs.get('loss_task', torch.tensor(0.0)).item()
            total_train_ponder_loss_epoch += outputs.get('loss_ponder_regularization', torch.tensor(0.0)).item()
            total_train_expected_steps_epoch += outputs.get('expected_ponder_cost_scalar', torch.tensor(0.0)).item()
            num_train_batches_epoch +=1

            if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == len(train_loader):
                if num_train_batches_epoch > 0 : # Avoid division by zero if log_interval is smaller than seen batches
                    current_avg_loss = total_train_loss_epoch / num_train_batches_epoch if (batch_idx +1) % args.log_interval != 0 else total_train_loss_epoch / args.log_interval
                    current_avg_task_loss = total_train_task_loss_epoch / num_train_batches_epoch if (batch_idx +1) % args.log_interval != 0 else total_train_task_loss_epoch / args.log_interval
                    current_avg_ponder_loss = total_train_ponder_loss_epoch / num_train_batches_epoch if (batch_idx +1) % args.log_interval != 0 else total_train_ponder_loss_epoch / args.log_interval
                    current_avg_expected_steps = total_train_expected_steps_epoch / num_train_batches_epoch if (batch_idx +1) % args.log_interval != 0 else total_train_expected_steps_epoch / args.log_interval
                    
                    progress_bar.set_postfix({
                        'Loss': f'{current_avg_loss:.4f}',
                        'TaskL': f'{current_avg_task_loss:.4f}',
                        'PondL': f'{current_avg_ponder_loss:.4f}',
                        'ExpSteps': f'{current_avg_expected_steps:.2f}'
                    })
                    if (batch_idx+1) % args.log_interval == 0 : # Reset for next interval if this was an interval boundary
                         total_train_loss_epoch, total_train_task_loss_epoch, total_train_ponder_loss_epoch, total_train_expected_steps_epoch = 0,0,0,0
                         num_train_batches_epoch = 0


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
                loss_val = outputs.get('loss_total')

                if loss_val is not None:
                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        print(f"Warning: NaN/Inf validation loss at epoch {epoch}. Value: {loss_val.item()}")
                        continue
                    total_val_loss += loss_val.item()
                    total_val_task_loss += outputs.get('loss_task', torch.tensor(0.0)).item()
                    total_val_ponder_loss += outputs.get('loss_ponder_regularization', torch.tensor(0.0)).item()
                    total_val_expected_steps += outputs.get('expected_ponder_cost_scalar', torch.tensor(0.0)).item()
                else:
                    print("Warning: Total loss not found during validation. Aggregating only steps.")
                    total_val_expected_steps += outputs.get('expected_ponder_cost_scalar', torch.tensor(0.0)).item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        avg_val_task_loss = total_val_task_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_ponder_loss = total_val_ponder_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_expected_steps = total_val_expected_steps / num_val_batches if num_val_batches > 0 else 0
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {epoch} Summary: Val_Loss: {avg_val_loss:.4f} | Val_TaskL: {avg_val_task_loss:.4f} | "
              f"Val_PondL(E[N]): {avg_val_ponder_loss:.4f} | Val_ExpSteps: {avg_val_expected_steps:.2f} | Time: {epoch_duration:.2f}s")

        if num_val_batches > 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                model_path = os.path.join(save_dir, f'best_model_epoch_{epoch}_valloss_{best_val_loss:.4f}.pt')
                torch.save(model.state_dict(), model_path)
                print(f"Saved new best model to {model_path}")
            except Exception as e:
                print(f"Error saving best model: {e}")
        
        try:
            latest_model_path = os.path.join(save_dir, 'latest_model.pt')
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                'args': serializable_args
            }, latest_model_path)
        except Exception as e:
            print(f"Error saving latest checkpoint: {e}")

    print("Training finished.")
    try:
        final_model_path = os.path.join(save_dir, 'final_model.pt')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

if __name__ == '__main__':
    args = get_args()
    if not args.disable_pos_enc and args.field_channels % 4 != 0 :
        print(f"ERROR: field_channels ({args.field_channels}) must be divisible by 4 for AddSinePositionalEncoding2D when positional encoding is enabled.")
        print("Please adjust --field_channels or use --disable_pos_enc.")
        exit()
    train(args)
