# Self-Supervised Pilot-Free Neural Demapping with 5G NR LDPC

**IEEE Access Submission — Hamail Ali & Shervin Vakili (Corresponding)**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Abstract.** We propose a pilot-free neural demapper that processes raw OFDM time-frequency resource grids and outputs calibrated soft Log-Likelihood Ratios (LLRs) for downstream 5G NR LDPC decoding. During deployment, the receiver continuously self-supervises via LDPC syndrome verification: syndrome-verified decoded codewords are admitted as pseudo-labels into an experience replay buffer, which drives online gradient updates of the ResCNN demapper — entirely without ground-truth transmitted bits. Compared to the standard 3GPP LS-estimated pilot-aided receiver, the proposed system achieves approximately **2 dB SNR gain** at the same post-FEC Block Error Rate target, attributable to the elimination of pilot overhead (14.3% resource savings) and avoidance of LS estimation noise.

---

## 📋 Contents

```
Self-Supervised-Pilot-Free-Neural-Demapping/
│
├── src/
│   ├── models.py            # ResCNN Demapper architecture (ConvBlock2D, ResBlock2D, ResCNNDemapper)
│   ├── channel_models.py    # AWGN, Rayleigh fading, DynamicChannelScheduler
│   ├── adaptation.py        # ExperienceReplayBuffer + OnlineSelfSupervisedAdaptation (Algorithm 1)
│   ├── baselines.py         # ExactLogMapAWGN, LSLMMSEBaseline, PerfectCSIMMSE
│   └── __init__.py
│
├── experiments/
│   ├── 01_pretrain_demapper.py         # Supervised pretraining on synthetic OFDM data
│   ├── 02_online_adaptation.py         # Main online self-supervised adaptation experiment
│   ├── 03_ablation_study.py            # 3-way ablation: Fixed vs. Self-sup. vs. Supervised
│   ├── 04_dynamic_channel_tracking.py  # BER recovery after sudden channel transition
│   └── 05_evaluate_baselines.py        # Full BER curves vs. all conventional baselines
│
├── checkpoints/
│   └── pretrained_demapper_16qam.pt    # Pretrained 16-QAM ResCNN checkpoint
│
├── results/                            # CSV outputs from experiments (auto-created)
│
├── paper/
│   ├── IEEE_Access_PilotFree_Neural_Demapper.tex   # LaTeX manuscript
│   ├── ieeeaccess.cls                               # IEEE Access class file
│   └── ber_master_results.png                       # Main BER figure
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🏗️ Architecture

```
Received OFDM Grid y ∈ ℂ^{B × SC × SYM}
         │
         ▼
   [Re(y) | Im(y)]  →  2 channels
         │
         ▼
   ConvBlock2D(2 → 64 ch)   GroupNorm + GELU
         │
         ▼
   ResBlock2D × 4           Skip-connected 3×3 Conv
         │
         ▼
   Conv2d(64 → M ch)        M = bits/symbol (4/6/8)
         │
         ▼
   Temperature Scaling (learnable τ)
         │
         ▼
   LLRs ∈ ℝ^{B × SC·SYM·M}
         │
         ▼
   5G NR LDPC BP Decoder
         │
         ├──→ Syndrome check  ──→  Pseudo-label admission  ──→  Experience Replay Buffer
         │                                                              │
         │                                                              ▼
         └──→ Decoded bits ───────────────────────────────→  Online SGD update (Adam)
```

### Online Self-Supervised Adaptation (Algorithm 1)

```
For each received OFDM block y:
  1. Forward pass: LLRs = ResCNN(y)
  2. Hard decisions: b_raw = sign(LLRs) > 0
  3. LDPC decode: (c_hat, syndrome) = LDPC_BP(LLRs)
  4. Quality gate 1: if syndrome ≠ 0 → reject
  5. Quality gate 2: if Hamming(b_raw, c_hat) / K > ε → reject   [ε = 0.12]
  6. Admit (y, c_hat) to experience replay buffer
  7. Every 32 admissions: sample mini-batch, compute BCE loss, gradient step
```

> **Note on pseudo-label reliability:** A zero syndrome indicates the decoded word is a valid codeword, but not necessarily the *transmitted* codeword. Gate 2 (Hamming disagreement ≤ 0.12) provides an additional empirical safeguard against undetected error events where the LDPC decoder converges to an incorrect valid codeword.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/HamailAli-INRS/Self-Supervised-Pilot-Free-Neural-Demapping.git
cd Self-Supervised-Pilot-Free-Neural-Demapping

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0.0 | Neural network training & inference |
| `numpy` | ≥ 1.24  | Numerical utilities |
| `scipy` | ≥ 1.11  | Jakes' Doppler model (`scipy.special.j0`) |
| `matplotlib` | ≥ 3.7 | BER curve plotting |
| `sionna` | 2.0.1 | (Optional) 5G NR LDPC encoder/decoder |

> **Without Sionna:** All channel simulation and baseline evaluation runs without Sionna. The LDPC decoder in `experiments/02_*` and `03_*` falls back to a minimal parity-check stub. For production LDPC decoding, install `sionna==2.0.1` and replace the stub with `sionna.phy.fec.ldpc.LDPC5GDecoder`.

---

## 🚀 Reproducing the Results

### Step 1 — Pretrain the ResCNN Demapper

```bash
python experiments/01_pretrain_demapper.py --mod 16 --epochs 50
python experiments/01_pretrain_demapper.py --mod 64 --epochs 50
```

Saves checkpoints to `checkpoints/pretrained_demapper_{mod}qam.pt`.  
A pretrained 16-QAM checkpoint is already provided in `checkpoints/`.

### Step 2 — Online Self-Supervised Adaptation (Figure 3 in paper)

```bash
python experiments/02_online_adaptation.py --mod 16 --snr_db 10 --channel awgn
python experiments/02_online_adaptation.py --mod 64 --snr_db 12 --channel rayleigh
```

### Step 3 — Ablation Study (Table II in paper)

Compares: **Fixed ResCNN** vs. **Self-Supervised (Proposed)** vs. **Supervised Upper Bound**.  
All three configs start from the **same pretrained checkpoint**.

```bash
python experiments/03_ablation_study.py --mod 16 --snr_db 10 --channel awgn
```

### Step 4 — Dynamic Channel Tracking (Figure 4 in paper)

Demonstrates BER recovery after sudden channel transition:
`AWGN (slots 0–1999)` → `Rayleigh (slots 2000–2999)` → `Rayleigh + 120 Hz Doppler (slots 3000–4999)`

```bash
python experiments/04_dynamic_channel_tracking.py --mod 16
```

### Step 5 — Full BER Curves vs. All Baselines (Figure 2 in paper)

```bash
python experiments/05_evaluate_baselines.py --mod 16 --channel awgn
python experiments/05_evaluate_baselines.py --mod 64 --channel rayleigh
```

> **On AWGN baseline fairness (Prof. Vakili's Point 5):** The `exact_logmap` column represents the theoretical optimal receiver (perfect CSI + exact Log-MAP LLR). The proposed neural receiver is compared against `ls_lmmse` (practical 3GPP LS-estimated pilot-aided receiver). The ~2 dB gain over `ls_lmmse` is due to pilot resource savings and avoidance of LS estimation error — **not** a gain over the information-theoretic bound.

---

## 📊 Key Results

| Configuration | 16-QAM @ 6 dB | 64-QAM @ 10 dB |
|--------------|--------------|---------------|
| Exact Log-MAP AWGN (bound) | ~1×10⁻⁴ | ~8×10⁻⁴ |
| LS+LMMSE 3GPP (practical) | ~6×10⁻³ | ~1×10⁻² |
| **Proposed Neural (pilot-free)** | **~1×10⁻³** | **~2×10⁻³** |
| Supervised upper bound | ~8×10⁻⁴ | ~9×10⁻⁴ |

*Post-FEC BER after 5G NR LDPC decoding (rate 1/2, codeword length 1024 bits).*

---

## 🔧 Key Configuration Parameters

| Parameter | Value | File | Notes |
|-----------|-------|------|-------|
| `epsilon_thresh` | **0.12** | `adaptation.py` | Disagreement gate (tightened from initial 0.30) |
| `update_interval` | 32 | `adaptation.py` | Gradient step every N admitted pseudo-labels |
| `hidden_dim` | 64 | `models.py` | ResCNN feature channels |
| `num_res_blocks` | 4 | `models.py` | Residual blocks in backbone |
| `lr` | 1×10⁻⁴ | `adaptation.py` | Online Adam learning rate |
| `buffer_capacity` | 2048 | `adaptation.py` | Experience replay buffer size |

---

## 🏢 Authors

**Hamail Ali**  
Dept. of Electrical Engineering, Military College of Signals (MCS), NUST  
Islamabad 44000, Pakistan  
📧 hamail1073@gmail.com

**Shervin Vakili** *(Corresponding Author)*  
ECCoLe Lab, INRS-EMT, Université du Québec  
Montréal, QC H5A 1K6, Canada  
📧 shervin.vakili@inrs.ca

---

## 🙏 Acknowledgments

This research was supported by the **Mitacs Globalink Research Internship (GRI) Program** and the **Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grant RGPIN-2022-04531**.

---

## 📄 Citation

```bibtex
@article{ali2026selfsuper,
  title   = {Self-Supervised Pilot-Free Neural Demapping With {5G NR LDPC}
             for High-Order Modulation Over {OFDM} Channels},
  author  = {Ali, Hamail and Vakili, Shervin},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Submitted}
}
```

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
