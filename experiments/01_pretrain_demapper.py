"""
Experiment 01 – Supervised Pretraining of ResCNN Demapper.

Trains the ResCNNDemapper on simulated AWGN and Rayleigh fading data
with ground-truth bit labels (supervised).  The resulting checkpoint
serves as the common initialization for ALL downstream experiments
(ablation study, online adaptation, dynamic channel tracking).

Usage
-----
    python experiments/01_pretrain_demapper.py [--mod {16,64,256}] [--epochs 50]

Output
------
    checkpoints/pretrained_demapper_{mod}qam.pt
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import ResCNNDemapper
from src.channel_models import AWGNChannel, RayleighFadingChannel


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def _gray_code(n: int) -> int:
    return n ^ (n >> 1)


def build_gray_qam(bits_per_symbol: int, device: torch.device) -> torch.Tensor:
    """Return Gray-coded QAM constellation symbols [2^M], unit power."""
    import math
    M   = bits_per_symbol
    num = 2 ** M
    side = int(math.isqrt(num))
    pam = torch.arange(side, dtype=torch.float32) * 2 - (side - 1)
    gray_pam = torch.zeros(side)
    for i in range(side):
        gray_pam[_gray_code(i)] = pam[i]
    real_part = gray_pam.repeat(side)
    imag_part = gray_pam.repeat_interleave(side)
    symbols   = torch.complex(real_part, imag_part)
    avg_power = (symbols.abs() ** 2).mean()
    return (symbols / avg_power.sqrt()).to(device)


def generate_batch(
    batch_size:      int,
    n_subcarriers:   int,
    n_symbols:       int,
    bits_per_symbol: int,
    snr_db_range:    tuple,
    channel_type:    str,
    device:          torch.device,
):
    """
    Generate one training batch.

    Returns
    -------
    y_grid    : [B, SC, SYM] complex64 — received OFDM grid
    bits      : [B, SC*SYM*M] float32  — transmitted bits (labels)
    """
    M   = bits_per_symbol
    num = 2 ** M
    N   = n_subcarriers * n_symbols

    const = build_gray_qam(M, device)                      # [2^M]

    # Random bit indices → constellation symbols
    idx   = torch.randint(0, num, (batch_size, N), device=device)
    x     = const[idx].view(batch_size, n_subcarriers, n_symbols)  # [B, SC, SYM]

    # Ground-truth bits via Gray mapping
    point_bits = []
    for i in range(num):
        point_bits.append([int(b) for b in format(i, f'0{M}b')])
    pb = torch.tensor(point_bits, dtype=torch.float32, device=device)  # [2^M, M]
    bits = pb[idx.view(-1)].view(batch_size, N * M)                     # [B, N*M]

    # Random SNR for each batch item within range
    snr_lo, snr_hi = snr_db_range
    snr_db = torch.rand(1).item() * (snr_hi - snr_lo) + snr_lo

    if channel_type == 'awgn':
        ch = AWGNChannel()
        y  = ch(x, snr_db)
    else:
        ch = RayleighFadingChannel(doppler_hz=0.0)
        y  = ch(x, snr_db)

    return y, bits


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def pretrain(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Pretrain] Device: {device}")
    print(f"[Pretrain] Modulation: {args.mod}-QAM | Epochs: {args.epochs}")

    M          = {16: 4, 64: 6, 256: 8}[args.mod]
    model      = ResCNNDemapper(num_bits_per_symbol=M, hidden_dim=64, num_res_blocks=4)
    model.to(device)
    print(f"[Pretrain] Parameters: {model.count_parameters():,}")

    optimizer  = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion  = nn.BCEWithLogitsLoss()

    n_sc, n_sym = 64, 14   # 5G NR numerology-0 slot: 64 subcarriers × 14 symbols
    snr_range   = (0.0, 20.0)

    best_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            y_grid, bits = generate_batch(
                batch_size=args.batch_size,
                n_subcarriers=n_sc,
                n_symbols=n_sym,
                bits_per_symbol=M,
                snr_db_range=snr_range,
                channel_type='awgn' if step % 2 == 0 else 'rayleigh',
                device=device,
            )

            optimizer.zero_grad()
            llrs = model(y_grid)                 # [B, SC*SYM*M]
            loss = criterion(llrs, bits)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / args.steps_per_epoch
        elapsed  = time.time() - t0

        print(f"  Epoch {epoch:3d}/{args.epochs} | Loss={avg_loss:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = f'checkpoints/pretrained_demapper_{args.mod}qam.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'num_bits_per_symbol': M,
                    'hidden_dim': 64,
                    'num_res_blocks': 4,
                    'modulation': f'{args.mod}-QAM',
                }
            }, ckpt_path)
            print(f"    ✓ Checkpoint saved → {ckpt_path}")

    print(f"\n[Pretrain] Done.  Best loss: {best_loss:.4f}")
    print(f"[Pretrain] Load with: torch.load('checkpoints/pretrained_demapper_{args.mod}qam.pt')")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain ResCNN Demapper')
    parser.add_argument('--mod',              type=int, default=16,  choices=[16, 64, 256],
                        help='QAM modulation order (default: 16)')
    parser.add_argument('--epochs',           type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size',       type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--steps_per_epoch',  type=int, default=200,
                        help='Gradient steps per epoch (default: 200)')
    args = parser.parse_args()
    pretrain(args)
