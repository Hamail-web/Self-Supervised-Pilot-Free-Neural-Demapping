"""
Experiment 04 – Dynamic Channel Tracking .

Demonstrates that the online self-supervised adaptation recovers BER after
a *sudden* channel transition during operation — without ground-truth labels.

Experiment design
-----------------
  Phase 0  (slots   0–1999): AWGN,  SNR = 12 dB
  Phase 1  (slots 2000–2999): Rayleigh (quasi-static), SNR = 12 dB  [channel onset]
  Phase 2  (slots 3000–4999): Rayleigh + Doppler (120 Hz), SNR = 12 dB  [mobility]

Comparison
----------
  Fixed ResCNN         – No online updates; BER degrades after transitions.
  Adaptive ResCNN (ours) – Online self-supervised; BER recovers.

Usage
-----
    python experiments/04_dynamic_channel_tracking.py [--mod {16,64}]

Output
------
    results/04_dynamic_channel_{mod}qam.csv
"""

import argparse
import csv
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import ResCNNDemapper
from src.channel_models import DynamicChannelScheduler
from src.adaptation import OnlineSelfSupervisedAdaptation


# ---------------------------------------------------------------------------
# Helpers
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
    sym = torch.complex(gp.repeat(side), gp.repeat_interleave(side))
    return (sym / (sym.abs() ** 2).mean().sqrt()).to(device)


def generate_slot_scheduler(M, n_sc, n_sym, scheduler, slot, device):
    """Generate one slot using the dynamic channel scheduler."""
    num  = 2 ** M
    N    = n_sc * n_sym
    c    = build_const(M, device)
    idx  = torch.randint(0, num, (1, N), device=device)
    x    = c[idx].view(1, n_sc, n_sym).squeeze(0)             # [SC, SYM]
    pb   = torch.tensor([[int(b) for b in format(i, f'0{M}b')] for i in range(num)],
                         dtype=torch.float32, device=device)
    bits = pb[idx.view(-1)].view(N * M)                        # [N*M]

    result = scheduler.transmit(x, slot)
    if len(result) == 2:
        y, label = result
    else:
        y, _, label = result

    return y, bits, label


def ldpc_decode_stub(llrs, parity_bits=48):
    hard = (llrs > 0).long()
    syn  = hard[:parity_bits].sum() % 2
    return hard, (syn.item() == 0)


def compute_ber(model, y, true_bits, device):
    model.eval()
    with torch.no_grad():
        llrs = model(y.unsqueeze(0).to(device)).squeeze(0).cpu()
    return ((llrs > 0).float() != true_bits.float()).float().mean().item()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    M      = {16: 4, 64: 6}[args.mod]

    ckpt_path = f'checkpoints/pretrained_demapper_{args.mod}qam.pt'
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    def load_model():
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt['config']
        m    = ResCNNDemapper(cfg['num_bits_per_symbol'], cfg['hidden_dim'], cfg['num_res_blocks'])
        m.load_state_dict(ckpt['model_state_dict'])
        return m.to(device)

    model_fixed    = load_model()
    model_adaptive = load_model()

    adapter = OnlineSelfSupervisedAdaptation(
        model=model_adaptive,
        lr=5e-5,
        update_interval=16,
        mini_batch_size=16,
        epsilon_thresh=0.12,
        device=device,
    )

    # --- Channel schedule ---
    schedule = [
        {'start_slot':    0, 'channel': 'awgn',     'snr_db': 12.0},
        {'start_slot': 2000, 'channel': 'rayleigh',  'snr_db': 12.0, 'doppler_hz':   0.0},
        {'start_slot': 3000, 'channel': 'rayleigh',  'snr_db': 12.0, 'doppler_hz': 120.0},
    ]
    scheduler = DynamicChannelScheduler(schedule)
    print(scheduler.get_schedule_summary())
    print(f"\n[Exp04] {args.mod}-QAM | {args.num_slots} slots | "
          f"Eval every {args.eval_interval} slots\n")

    n_sc, n_sym = 64, 14
    os.makedirs('results', exist_ok=True)
    csv_path = f'results/04_dynamic_channel_{args.mod}qam.csv'
    results  = []

    print(f"{'Slot':>6}  {'Phase':>12}  {'BER-Fixed':>10}  {'BER-Adaptive':>13}")
    print("-" * 50)

    torch.manual_seed(0)
    for slot in range(args.num_slots):
        y, true_bits, ch_label = generate_slot_scheduler(M, n_sc, n_sym, scheduler, slot, device)

        # Adaptive model: pseudo-label step
        model_adaptive.eval()
        with torch.no_grad():
            llrs = model_adaptive(y.unsqueeze(0).to(device)).squeeze(0)
        dec_bits, syn_ok = ldpc_decode_stub(llrs.cpu())
        adapter.step(y, dec_bits, syn_ok, true_bits=true_bits)

        if (slot + 1) % args.eval_interval == 0:
            ber_f = compute_ber(model_fixed,    y, true_bits, device)
            ber_a = compute_ber(model_adaptive, y, true_bits, device)

            # Determine phase label
            if slot < 2000:
                phase = 'AWGN'
            elif slot < 3000:
                phase = 'Rayleigh'
            else:
                phase = 'Rayleigh+Doppler'

            print(f"{slot+1:>6}  {phase:>12}  {ber_f:>10.5f}  {ber_a:>13.5f}")
            results.append({
                'slot':         slot + 1,
                'phase':        phase,
                'ber_fixed':    ber_f,
                'ber_adaptive': ber_a,
            })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['slot', 'phase', 'ber_fixed', 'ber_adaptive'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Exp04] Results saved to {csv_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic Channel Tracking')
    parser.add_argument('--mod',           type=int, default=16, choices=[16, 64])
    parser.add_argument('--num_slots',     type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=50)
    args = parser.parse_args()
    run(args)
