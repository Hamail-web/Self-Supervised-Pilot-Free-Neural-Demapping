"""
Online Self-Supervised Adaptation via LDPC Syndrome Verification.

Implements Algorithm 1 from the paper:
  "Self-Supervised Pilot-Free Neural Demapping With 5G NR LDPC
   for High-Order Modulation Over OFDM Channels"

Key design choices (addressing Prof. Vakili's feedback):
  - Disagreement threshold tightened to epsilon_thresh = 0.12  (was 0.30)
    to admit only high-confidence pseudo-labels into the replay buffer.
  - Pseudo-label accuracy is tracked empirically (pseudo_label_accuracy_log)
    whenever ground-truth bits are available (ablation / supervised mode).
  - The adaptation starts from a *pretrained* model checkpoint rather than
    random initialization.  Online updates adapt the model continuously
    without requiring ground-truth labels during deployment.

References
----------
  Richardson & Urbanke (2008) – Modern Coding Theory (syndrome definition)
  Simeone (2022) – An Introduction to Machine Learning for Communications
"""

import collections
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------

class ExperienceReplayBuffer:
    """
    Fixed-capacity circular buffer of (y_grid, pseudo_label_bits) pairs.

    Pseudo-labels are admitted only when two quality gates pass:
      1. The LDPC decoder finds **zero syndrome** (the decoded word is a valid
         codeword under the parity-check matrix H: H·c ≡ 0 mod 2).
      2. The Hamming disagreement between raw hard decisions and the decoded
         bits does NOT exceed `epsilon_thresh` (fraction of bits).

    Gate 1 rules out burst-corrupted blocks.  Gate 2 rules out blocks where
    the decoder made a large number of corrections that may push the solution
    towards a *different* valid codeword (an undetected error event).

    Note on undetected-error probability
    -------------------------------------
    A zero syndrome is *necessary* but not *sufficient* for a correctly decoded
    codeword.  For 5G NR LDPC (rate 1/3–8/9) the undetected-error probability
    is negligible at operational SNR, but we do not make a specific numeric
    claim here.  Gate 2 provides an additional empirical safeguard.

    Parameters
    ----------
    capacity       : Maximum number of (y_grid, bits) pairs to retain.
    epsilon_thresh : Maximum allowed Hamming disagreement fraction [0, 1].
                     Set to 0.12 (Vakili feedback: tighten from 0.30).
    """

    def __init__(self, capacity: int = 2048, epsilon_thresh: float = 0.12):
        self.capacity        = capacity
        self.epsilon_thresh  = epsilon_thresh
        self._buffer: collections.deque = collections.deque(maxlen=capacity)
        self.total_offered   = 0
        self.total_admitted  = 0

    # ------------------------------------------------------------------
    def try_admit(
        self,
        y_grid: torch.Tensor,
        raw_hard_bits: torch.Tensor,
        decoded_bits:  torch.Tensor,
        syndrome_zero: bool,
    ) -> bool:
        """
        Attempt to admit one sample into the buffer.

        Parameters
        ----------
        y_grid        : Received OFDM grid [Subcarriers, Symbols] (complex64).
        raw_hard_bits : Hard decisions BEFORE LDPC decoding, shape [K].
        decoded_bits  : LDPC-decoded bits, shape [K].
        syndrome_zero : True if H·decoded_bits ≡ 0 (mod 2).

        Returns
        -------
        bool : True if the sample was admitted.
        """
        self.total_offered += 1

        # Gate 1 – LDPC syndrome check
        if not syndrome_zero:
            return False

        # Gate 2 – Hamming disagreement check
        disagreement = (raw_hard_bits != decoded_bits).float().mean().item()
        if disagreement > self.epsilon_thresh:
            return False

        # Both gates passed → store (detached CPU tensors to save VRAM)
        self._buffer.append((
            y_grid.detach().cpu(),
            decoded_bits.detach().cpu(),
        ))
        self.total_admitted += 1
        return True

    # ------------------------------------------------------------------
    def sample(self, batch_size: int):
        """
        Draw a random mini-batch from the buffer.

        Returns
        -------
        y_grids   : [batch_size, Subcarriers, Symbols] complex64
        bit_labels: [batch_size, K] float32
        or None if the buffer is too small.
        """
        if len(self._buffer) < batch_size:
            return None, None

        samples = random.sample(list(self._buffer), batch_size)
        y_grids    = torch.stack([s[0] for s in samples], dim=0)
        bit_labels = torch.stack([s[1] for s in samples], dim=0).float()
        return y_grids, bit_labels

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def admission_rate(self) -> float:
        """Fraction of offered samples that passed quality gates."""
        if self.total_offered == 0:
            return 0.0
        return self.total_admitted / self.total_offered


# ---------------------------------------------------------------------------
# Online Self-Supervised Adaptation Engine
# ---------------------------------------------------------------------------

class OnlineSelfSupervisedAdaptation:
    """
    Continually fine-tunes a *pretrained* ResCNN demapper using LDPC-verified
    pseudo-labels without requiring ground-truth transmitted bits.

    Algorithm (Algorithm 1 in the paper)
    -------------------------------------
    For each received OFDM block y:
      1. Run ResCNN → LLRs → hard decisions b_raw.
      2. Run 5G NR LDPC BP decoder → decoded codeword c_hat.
      3. Check syndrome: s = H · c_hat (mod 2).  If s ≠ 0 → discard.
      4. Check disagreement: d = hamming(b_raw, c_hat) / K.
         If d > ε_thresh → discard.
      5. Admit (y, c_hat) to replay buffer.
      6. Every `update_interval` slots, sample a mini-batch from the buffer
         and perform one gradient step minimising BCE(LLR, c_hat).

    Parameters
    ----------
    model          : ResCNNDemapper instance (pretrained).
    lr             : Learning rate for online Adam updates.
    update_interval: Gradient step every N admitted samples.
    mini_batch_size: Replay buffer mini-batch size.
    buffer_capacity: Maximum replay buffer size.
    epsilon_thresh : Disagreement threshold (default 0.12 per Vakili feedback).
    device         : torch.device.
    """

    def __init__(
        self,
        model:           nn.Module,
        lr:              float = 1e-4,
        update_interval: int   = 32,
        mini_batch_size: int   = 16,
        buffer_capacity: int   = 2048,
        epsilon_thresh:  float = 0.12,
        device:          Optional[torch.device] = None,
    ):
        self.model           = model
        self.device          = device or torch.device('cpu')
        self.update_interval = update_interval
        self.mini_batch_size = mini_batch_size

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

        self.buffer = ExperienceReplayBuffer(
            capacity=buffer_capacity,
            epsilon_thresh=epsilon_thresh,
        )

        # Logging / diagnostics
        self.slots_processed          = 0
        self.gradient_steps           = 0
        self.pseudo_label_accuracy_log: list[float] = []  # filled when true bits known
        self.loss_log:                  list[float] = []

    # ------------------------------------------------------------------
    def step(
        self,
        y_grid:         torch.Tensor,
        decoded_bits:   torch.Tensor,
        syndrome_zero:  bool,
        true_bits:      Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Process one received OFDM block.

        Parameters
        ----------
        y_grid        : Complex received grid [Subcarriers, Symbols].
        decoded_bits  : LDPC-decoded bits [K] (from external LDPC decoder).
        syndrome_zero : True if syndrome check passed.
        true_bits     : (Optional) ground-truth bits for accuracy tracking
                        (available in supervised ablation; not available in
                        deployment).

        Returns
        -------
        info : dict with keys:
            'admitted'           – bool
            'buffer_size'        – int
            'pseudo_label_acc'   – float | None  (only if true_bits provided)
            'loss'               – float | None  (if a gradient step was taken)
        """
        self.slots_processed += 1

        # --- Compute raw hard decisions from current model (no grad) ---
        self.model.eval()
        with torch.no_grad():
            y_in = y_grid.unsqueeze(0).to(self.device)        # [1, SC, SYM]
            llrs = self.model(y_in)                            # [1, K]
            raw_hard_bits = (llrs.squeeze(0) > 0).long().cpu()

        # --- Try to admit to replay buffer ---
        admitted = self.buffer.try_admit(
            y_grid        = y_grid,
            raw_hard_bits = raw_hard_bits,
            decoded_bits  = decoded_bits.cpu(),
            syndrome_zero = syndrome_zero,
        )

        # --- Optional: measure pseudo-label accuracy vs. true bits ---
        pseudo_acc = None
        if admitted and true_bits is not None:
            pseudo_acc = (decoded_bits.cpu() == true_bits.cpu()).float().mean().item()
            self.pseudo_label_accuracy_log.append(pseudo_acc)

        # --- Gradient update every `update_interval` admissions ---
        loss_val = None
        if self.buffer.total_admitted % self.update_interval == 0 and self.buffer.total_admitted > 0:
            loss_val = self._gradient_step()

        return {
            'admitted':         admitted,
            'buffer_size':      len(self.buffer),
            'pseudo_label_acc': pseudo_acc,
            'loss':             loss_val,
        }

    # ------------------------------------------------------------------
    def _gradient_step(self) -> Optional[float]:
        """Draw a mini-batch from the replay buffer and take one gradient step."""
        y_grids, bit_labels = self.buffer.sample(self.mini_batch_size)
        if y_grids is None:
            return None

        self.model.train()
        y_grids    = y_grids.to(self.device)
        bit_labels = bit_labels.to(self.device)

        self.optimizer.zero_grad()
        llrs = self.model(y_grids)   # [B, K]
        loss = self.criterion(llrs, bit_labels)
        loss.backward()
        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_log.append(loss_val)
        self.gradient_steps += 1
        return loss_val

    # ------------------------------------------------------------------
    def get_diagnostics(self) -> dict:
        """Return a summary of adaptation statistics."""
        avg_pseudo_acc = (
            sum(self.pseudo_label_accuracy_log) / len(self.pseudo_label_accuracy_log)
            if self.pseudo_label_accuracy_log else None
        )
        avg_loss = (
            sum(self.loss_log[-100:]) / len(self.loss_log[-100:])
            if self.loss_log else None
        )
        return {
            'slots_processed':      self.slots_processed,
            'gradient_steps':       self.gradient_steps,
            'buffer_size':          len(self.buffer),
            'buffer_admission_rate': self.buffer.admission_rate,
            'avg_pseudo_label_acc': avg_pseudo_acc,
            'avg_recent_loss':      avg_loss,
        }
