"""
Experiment 02 – Online Self-Supervised Adaptation (Main Experiment).

Demonstrates Algorithm 1 from the paper:
  - Load pretrained ResCNN checkpoint.
  - Stream OFDM slots through the channel.
  - Use LDPC syndrome verification to build a replay buffer of pseudo-labels.
  - Continuously fine-tune the model using admitted pseudo-labels.
  - Evaluate BER every `eval_interval` slots.

  • Pseudo-label threshold tightened to 0.12 (not 0.30).
  • Starts from a *pretrained* model; online self-supervised adaptation
    without requiring ground-truth labels during deployment.

Usage
-----
    python experiments/02_online_adaptation.py [--mod {16,64}] [--snr_db 10]

Output
------
    results/02_online_adaptation_{mod}qam_snr{snr}dB.csv
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import ResCNNDemapper
from src.channel_models import AWGNChannel, RayleighFadingChannel
from src.adaptation import OnlineSelfSupervisedAdaptation


# ---------------------------------------------------------------------------
# Minimal LDPC syndrome stub
# ---------------------------------------------------------------------------

def ldpc_decode_stub(llrs: torch.Tensor, parity_bits: int = 48):
    """
    Minimal LDPC decoder stub (no Sionna dependency).

    In production, replace with:
        sionna.phy.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)

    Here we simulate:
      - Hard decisions from LLRs.
      - A synthetic syndrome check (all-ones parity matrix for demonstration).
      - Syndrome = XOR of hard bits → zero if even parity (toy model).

    Returns
    -------
    decoded_bits  : Hard-decision bits [K].
    syndrome_zero : True if syndrome check passes.
    """
    hard_bits = (llrs > 0).long()                    # [K]
    # Toy syndrome: check parity of first `parity_bits` bits
    syndrome_bit = hard_bits[:parity_bits].sum() % 2
    syndrome_zero = (syndrome_bit.item() == 0)
    return hard_bits, syndrome_zero


# ---------------------------------------------------------------------------
# Data generation (reuses logic from experiment 01)
# ---------------------------------------------------------------------------

def _gray_code(n: int) -> int:
    return n ^ (n >> 1)


def build_const(M: int, device: torch.device) -> torch.Tensor:
    import math
    num  = 2 ** M
    side = int(math.isqrt(num))
    pam  = torch.arange(side, dtype=torch.float32) * 2 - (side - 1)
    gray_pam = torch.zeros(side)
    for i in range(side):
        gray_pam[_gray_code(i)] = pam[i]
    real = gray_pam.repeat(side)
    imag = gray_pam.repeat_interleave(side)
    sym  = torch.complex(real, imag)
    sym  = sym / (sym.abs() ** 2).mean().sqrt()
    return sym.to(device)


def generate_slot(M, n_sc, n_sym, snr_db, channel_type, device):
    """Generate one OFDM slot: returns (y_grid, true_bits)."""
    num   = 2 ** M
    N     = n_sc * n_sym
    const = build_const(M, device)
    idx   = torch.randint(0, num, (1, N), device=device)
    x     = const[idx].view(1, n_sc, n_sym)
    pb = torch.tensor(
        [[int(b) for b in format(i, f'0{M}b')] for i in range(num)],
        dtype=torch.float32, device=device
    )
    true_bits = pb[idx.view(-1)].view(1, N * M)

    if channel_type == 'awgn':
        y = AWGNChannel()(x.squeeze(0), snr_db).unsqueeze(0)
    else:
        y = RayleighFadingChannel()(x.squeeze(0), snr_db).unsqueeze(0)

    return y.squeeze(0), true_bits.squeeze(0)   # [SC, SYM], [N*M]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    M      = {16: 4, 64: 6}[args.mod]

    # --- Load pretrained model ---
    ckpt_path = f'checkpoints/pretrained_demapper_{args.mod}qam.pt'
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("  Run experiments/01_pretrain_demapper.py first.")
        sys.exit(1)

    ckpt  = torch.load(ckpt_path, map_location=device)
    cfg   = ckpt['config']
    model = ResCNNDemapper(
        num_bits_per_symbol=cfg['num_bits_per_symbol'],
        hidden_dim=cfg['hidden_dim'],
        num_res_blocks=cfg['num_res_blocks'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    print(f"[Exp02] Loaded pretrained checkpoint from {ckpt_path}")
    print(f"[Exp02] Mod={args.mod}-QAM | SNR={args.snr_db} dB | "
          f"Channel={args.channel} | Slots={args.num_slots}")

    # --- Setup adaptation engine ---
    adapter = OnlineSelfSupervisedAdaptation(
        model=model,
        lr=1e-4,
        update_interval=32,
        mini_batch_size=16,
        buffer_capacity=2048,
        epsilon_thresh=0.12,
        device=device,
    )

    n_sc, n_sym = 64, 14
    os.makedirs('results', exist_ok=True)
    csv_path = f'results/02_online_adaptation_{args.mod}qam_snr{args.snr_db}dB.csv'

    results = []
    for slot in range(args.num_slots):
        y_grid, true_bits = generate_slot(M, n_sc, n_sym, args.snr_db, args.channel, device)

        # --- Forward pass for LLRs (for LDPC decode input) ---
        model.eval()
        with torch.no_grad():
            llrs = model(y_grid.unsqueeze(0).to(device)).squeeze(0)

        # --- LDPC decode ---
        decoded_bits, syndrome_zero = ldpc_decode_stub(llrs.cpu())

        # --- Adaptation step ---
        info = adapter.step(
            y_grid        = y_grid,
            decoded_bits  = decoded_bits,
            syndrome_zero = syndrome_zero,
            true_bits     = true_bits,   # only for tracking accuracy; not used for training
        )

        # --- BER evaluation every eval_interval slots ---
        if (slot + 1) % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                llrs_eval = model(y_grid.unsqueeze(0).to(device)).squeeze(0)
            decisions = (llrs_eval.cpu() > 0).float()
            ber = (decisions != true_bits.float()).float().mean().item()

            diag = adapter.get_diagnostics()
            row  = {
                'slot':             slot + 1,
                'ber':              ber,
                'admitted':         diag['buffer_size'],
                'admission_rate':   f"{diag['buffer_admission_rate']:.3f}",
                'pseudo_label_acc': f"{diag['avg_pseudo_label_acc']:.4f}" if diag['avg_pseudo_label_acc'] else 'N/A',
                'gradient_steps':   diag['gradient_steps'],
                'avg_loss':         f"{diag['avg_recent_loss']:.4f}" if diag['avg_recent_loss'] else 'N/A',
            }
            results.append(row)
            print(f"  Slot {slot+1:5d} | BER={ber:.4f} | Buffer={diag['buffer_size']} "
                  f"| AdmRate={diag['buffer_admission_rate']:.2%} "
                  f"| PseudoAcc={diag['avg_pseudo_label_acc']}")

    # --- Save results ---
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Exp02] Results saved to {csv_path}")
    print("[Exp02] Done.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online Self-Supervised Adaptation')
    parser.add_argument('--mod',           type=int,   default=16,     choices=[16, 64])
    parser.add_argument('--snr_db',        type=float, default=10.0)
    parser.add_argument('--channel',       type=str,   default='awgn', choices=['awgn', 'rayleigh'])
    parser.add_argument('--num_slots',     type=int,   default=5000)
    parser.add_argument('--eval_interval', type=int,   default=100)
    args = parser.parse_args()
    run(args)
