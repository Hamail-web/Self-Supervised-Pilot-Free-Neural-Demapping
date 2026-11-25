"""
Experiment 03 – Ablation Study (Addresses Prof. Vakili's Point 3).

Compares three configurations, all starting from the same pretrained checkpoint:

  Config A – Fixed ResCNN:
    Pretrained model, NO online weight updates (frozen during deployment).

  Config B – ResCNN + Pseudo-label Learning + Replay (Proposed):
    Online self-supervised adaptation via LDPC syndrome-verified pseudo-labels
    with experience replay (Algorithm 1 in the paper).

  Config C – Supervised ResCNN (Upper Bound):
    Online adaptation using the TRUE transmitted bits as labels (not available
    during deployment; serves as the theoretical upper bound for Config B).

All three configurations:
  - Start from the identical pretrained checkpoint.
  - Use the same received OFDM signal sequence (same random seed).
  - Use the same adaptation budget (same update_interval, mini_batch_size).

Usage
-----
    python experiments/03_ablation_study.py [--mod {16,64}] [--snr_db 10]

Output
------
    results/03_ablation_{mod}qam_snr{snr}dB.csv
"""

import argparse
import csv
import copy
import math
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import ResCNNDemapper
from src.channel_models import AWGNChannel, RayleighFadingChannel
from src.adaptation import OnlineSelfSupervisedAdaptation, ExperienceReplayBuffer


# ---------------------------------------------------------------------------
# Helpers (re-used from experiment 01/02)
# ---------------------------------------------------------------------------

def _gray_code(n):
    return n ^ (n >> 1)


def build_const(M, device):
    num  = 2 ** M
    side = int(math.isqrt(num))
    pam  = torch.arange(side, dtype=torch.float32) * 2 - (side - 1)
    gp   = torch.zeros(side)
    for i in range(side):
        gp[_gray_code(i)] = pam[i]
    real = gp.repeat(side)
    imag = gp.repeat_interleave(side)
    sym  = torch.complex(real, imag)
    return (sym / (sym.abs() ** 2).mean().sqrt()).to(device)


def generate_slot(M, n_sc, n_sym, snr_db, channel_type, device):
    num  = 2 ** M
    N    = n_sc * n_sym
    c    = build_const(M, device)
    idx  = torch.randint(0, num, (1, N), device=device)
    x    = c[idx].view(1, n_sc, n_sym)
    pb   = torch.tensor([[int(b) for b in format(i, f'0{M}b')] for i in range(num)],
                         dtype=torch.float32, device=device)
    bits = pb[idx.view(-1)].view(1, N * M)
    if channel_type == 'awgn':
        y = AWGNChannel()(x.squeeze(0), snr_db).unsqueeze(0)
    else:
        y = RayleighFadingChannel()(x.squeeze(0), snr_db).unsqueeze(0)
    return y.squeeze(0), bits.squeeze(0)


def ldpc_decode_stub(llrs, parity_bits=48):
    hard_bits = (llrs > 0).long()
    syndrome_bit = hard_bits[:parity_bits].sum() % 2
    return hard_bits, (syndrome_bit.item() == 0)


def compute_ber(model, y_grid, true_bits, device):
    model.eval()
    with torch.no_grad():
        llrs = model(y_grid.unsqueeze(0).to(device)).squeeze(0).cpu()
    return (((llrs > 0).float()) != true_bits.float()).float().mean().item()


# ---------------------------------------------------------------------------
# Config C: Supervised online adaptation (upper bound)
# ---------------------------------------------------------------------------

class SupervisedOnlineAdapter:
    """Adapts the model using TRUE labels — not available in deployment."""

    def __init__(self, model, lr=1e-4, update_interval=32, batch_size=16, device=None):
        self.model    = model
        self.device   = device or torch.device('cpu')
        self.interval = update_interval
        self.bs       = batch_size
        self._buffer  = []
        self._step    = 0
        self.opt      = optim.Adam(model.parameters(), lr=lr)
        self.crit     = nn.BCEWithLogitsLoss()

    def step(self, y_grid, true_bits):
        self._buffer.append((y_grid.cpu(), true_bits.cpu()))
        if len(self._buffer) > 2048:
            self._buffer.pop(0)
        self._step += 1
        if self._step % self.interval == 0 and len(self._buffer) >= self.bs:
            import random
            batch = random.sample(self._buffer, self.bs)
            yb = torch.stack([s[0] for s in batch]).to(self.device)
            bb = torch.stack([s[1] for s in batch]).float().to(self.device)
            self.model.train()
            self.opt.zero_grad()
            loss = self.crit(self.model(yb), bb)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    M      = {16: 4, 64: 6}[args.mod]

    ckpt_path = f'checkpoints/pretrained_demapper_{args.mod}qam.pt'
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("  Run 01_pretrain_demapper.py first.")
        sys.exit(1)

    def load_model():
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt['config']
        m    = ResCNNDemapper(cfg['num_bits_per_symbol'], cfg['hidden_dim'], cfg['num_res_blocks'])
        m.load_state_dict(ckpt['model_state_dict'])
        return m.to(device)

    model_A = load_model()                       # Config A: Fixed (no updates)
    model_B = load_model()                       # Config B: Self-supervised (proposed)
    model_C = load_model()                       # Config C: Supervised upper bound

    adapter_B = OnlineSelfSupervisedAdaptation(
        model=model_B, lr=1e-4, update_interval=32,
        mini_batch_size=16, epsilon_thresh=0.12, device=device,
    )
    adapter_C = SupervisedOnlineAdapter(
        model=model_C, lr=1e-4, update_interval=32,
        batch_size=16, device=device,
    )

    n_sc, n_sym = 64, 14
    os.makedirs('results', exist_ok=True)
    csv_path = f'results/03_ablation_{args.mod}qam_snr{args.snr_db}dB.csv'
    results  = []

    print(f"[Exp03] Ablation | {args.mod}-QAM | SNR={args.snr_db} dB | "
          f"Channel={args.channel} | Slots={args.num_slots}")
    print(f"{'Slot':>6}  {'BER-A(Fixed)':>14}  {'BER-B(Self-sup)':>16}  {'BER-C(Supervised)':>18}")
    print("-" * 65)

    torch.manual_seed(42)   # Same data for all three configs
    for slot in range(args.num_slots):
        y_grid, true_bits = generate_slot(M, n_sc, n_sym, args.snr_db, args.channel, device)

        # Config B: pseudo-label adaptation
        with torch.no_grad():
            llrs_b = model_B(y_grid.unsqueeze(0).to(device)).squeeze(0)
        dec_bits, syn_ok = ldpc_decode_stub(llrs_b.cpu())
        adapter_B.step(y_grid, dec_bits, syn_ok, true_bits=true_bits)

        # Config C: supervised adaptation
        adapter_C.step(y_grid, true_bits)

        if (slot + 1) % args.eval_interval == 0:
            ber_a = compute_ber(model_A, y_grid, true_bits, device)
            ber_b = compute_ber(model_B, y_grid, true_bits, device)
            ber_c = compute_ber(model_C, y_grid, true_bits, device)
            print(f"{slot+1:>6}  {ber_a:>14.5f}  {ber_b:>16.5f}  {ber_c:>18.5f}")
            results.append({'slot': slot+1, 'ber_fixed': ber_a,
                             'ber_self_supervised': ber_b, 'ber_supervised': ber_c})

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['slot', 'ber_fixed', 'ber_self_supervised', 'ber_supervised'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Exp03] Results saved to {csv_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--mod',           type=int,   default=16,     choices=[16, 64])
    parser.add_argument('--snr_db',        type=float, default=10.0)
    parser.add_argument('--channel',       type=str,   default='awgn', choices=['awgn', 'rayleigh'])
    parser.add_argument('--num_slots',     type=int,   default=5000)
    parser.add_argument('--eval_interval', type=int,   default=100)
    args = parser.parse_args()
    run(args)
