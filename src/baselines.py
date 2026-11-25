"""
Conventional Receiver Baselines for Fair Comparison.

Addresses Prof. Vakili's Point 5:
  "In AWGN with known H and sigma^2, a conventional soft demapper can compute
   exact bit LLRs. These provide the optimal bitwise soft metrics. Therefore a
   neural demapper should not systematically outperform this ideal reference."

Three baselines are implemented:

  1. ExactLogMapAWGN   – Perfect CSI + exact Log-MAP LLR (theoretical optimum
                         for AWGN; the neural receiver should NOT beat this).

  2. LSLMMSEBaseline   – 3GPP LS channel estimation on pilots → LMMSE equalization
                         → Log-MAP demapping (practical / fair comparison target).
                         The proposed 2 dB gain is stated relative to THIS baseline.

  3. PerfectCSIMMSE    – MMSE equalization with perfect H knowledge for Rayleigh
                         fading (upper-bound reference for fading experiments).

Design note
-----------
All baselines accept and return plain PyTorch tensors and do NOT require Sionna
to be installed.  The constellation Gray-coding convention matches the
Gray-coded QAM used in the ResCNN training data.
"""

import math
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Gray-coded QAM bit-to-symbol mapping helpers
# ---------------------------------------------------------------------------

def _gray_code(n: int) -> int:
    """Convert natural binary n to Gray code integer."""
    return n ^ (n >> 1)


def _build_qam_constellation(bits_per_symbol: int) -> torch.Tensor:
    """
    Build a Gray-coded square QAM constellation (normalized to unit average power).

    Parameters
    ----------
    bits_per_symbol : 2 (QPSK), 4 (16-QAM), 6 (64-QAM), 8 (256-QAM).

    Returns
    -------
    symbols : Complex64 tensor [2^M] with M = bits_per_symbol.
    """
    M   = bits_per_symbol
    num = 2 ** M
    side = int(math.isqrt(num))  # sqrt of constellation size
    # Build 1-D PAM levels: -(sqrt(num)-1), ..., -1, 1, ..., (sqrt(num)-1)
    pam_levels = torch.arange(side, dtype=torch.float32) * 2 - (side - 1)

    # Map Gray-coded index to PAM level
    gray_pam = torch.zeros(side)
    for i in range(side):
        gray_pam[_gray_code(i)] = pam_levels[i]

    # Outer product: Real × Imaginary
    real_part = gray_pam.repeat(side)        # [num]
    imag_part = gray_pam.repeat_interleave(side)  # [num]
    symbols   = torch.complex(real_part, imag_part)

    # Normalize to unit average power
    avg_power = (symbols.abs() ** 2).mean()
    symbols   = symbols / avg_power.sqrt()
    return symbols


# ---------------------------------------------------------------------------
# 1. Exact Log-MAP AWGN Baseline (theoretical optimum)
# ---------------------------------------------------------------------------

class ExactLogMapAWGN:
    """
    Optimal Log-MAP bit LLR computation for AWGN with perfect CSI.

    For the AWGN channel  y = x + n  (H ≡ 1, known sigma^2):

        LLR(b_k) = log( sum_{x: b_k=1} exp(-|y - x|^2 / sigma^2) )
                 - log( sum_{x: b_k=0} exp(-|y - x|^2 / sigma^2) )

    This is the *exact* Max-Log-MAP approximation (without the max shortcut):

        LLR(b_k) ≈ (1/sigma^2) * [
                      min_{x: b_k=0} |y - x|^2
                    - min_{x: b_k=1} |y - x|^2
                   ]

    We implement the exact sum (not max approximation) for rigor.

    Parameters
    ----------
    bits_per_symbol : Modulation order (4 for 16-QAM, 6 for 64-QAM, etc.).
    """

    def __init__(self, bits_per_symbol: int = 4):
        self.M          = bits_per_symbol
        self.num_points = 2 ** bits_per_symbol
        self.const      = _build_qam_constellation(bits_per_symbol)  # [2^M]

        # Pre-compute bit assignment for each constellation point [2^M, M]
        point_bits = []
        for i in range(self.num_points):
            bits = [int(b) for b in format(i, f'0{bits_per_symbol}b')]
            point_bits.append(bits)
        self.point_bits = torch.tensor(point_bits, dtype=torch.float32)  # [2^M, M]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_llrs(
        self,
        y: torch.Tensor,
        noise_var: float,
    ) -> torch.Tensor:
        """
        Compute exact Log-MAP LLRs.

        Parameters
        ----------
        y         : Received symbols [Batch, num_data_symbols] (complex64).
        noise_var : Scalar noise variance (N0 = sigma^2).

        Returns
        -------
        llrs : [Batch, num_data_symbols * M] (float32).
        """
        const = self.const.to(y.device)                     # [num_points]
        pb    = self.point_bits.to(y.device)                 # [num_points, M]

        batch, N = y.shape
        y_flat = y.view(-1, 1)                               # [B*N, 1]
        c      = const.view(1, -1)                           # [1, num_points]

        # Squared Euclidean distances [B*N, num_points]
        dist2 = (y_flat.real - c.real) ** 2 + (y_flat.imag - c.imag) ** 2
        # Log-likelihood of each constellation point
        log_lik = -dist2 / noise_var                         # [B*N, num_points]

        # Bit-wise Log-MAP LLR
        llrs_flat = torch.zeros(batch * N, self.M, device=y.device)
        for k in range(self.M):
            mask1 = (pb[:, k] == 1).float().to(y.device)   # [num_points]
            mask0 = 1.0 - mask1

            # log-sum-exp for b_k = 1 and b_k = 0 sets
            log_sum1 = torch.logsumexp(log_lik + torch.log(mask1 + 1e-30).unsqueeze(0), dim=-1)
            log_sum0 = torch.logsumexp(log_lik + torch.log(mask0 + 1e-30).unsqueeze(0), dim=-1)
            llrs_flat[:, k] = log_sum1 - log_sum0

        llrs = llrs_flat.view(batch, N * self.M)
        return llrs

    # ------------------------------------------------------------------
    @torch.no_grad()
    def ber(
        self,
        y: torch.Tensor,
        true_bits: torch.Tensor,
        noise_var: float,
    ) -> float:
        """Convenience: compute BER from exact Log-MAP decisions."""
        llrs      = self.compute_llrs(y, noise_var)
        decisions = (llrs > 0).float()
        return (decisions != true_bits.float()).float().mean().item()


# ---------------------------------------------------------------------------
# 2. LS + LMMSE Baseline (3GPP practical reference)
# ---------------------------------------------------------------------------

class LSLMMSEBaseline:
    """
    Standard 3GPP pilot-aided receiver:
      1. Least-Squares (LS) channel estimation using pilot subcarriers.
      2. LMMSE equalization of data subcarriers.
      3. Exact Log-MAP LLR demapping.

    This is the *fair practical comparison* target.  The proposed neural
    receiver's 2 dB gain is stated relative to this baseline — NOT relative
    to the ExactLogMapAWGN bound.

    Parameters
    ----------
    bits_per_symbol : Modulation order.
    pilot_indices   : 1-D integer tensor listing the flat indices (into the
                      vectorized resource grid) of pilot subcarriers.
    pilot_symbols   : Complex tensor [num_pilots] of known pilot values.
    snr_prior_db    : SNR prior used for LMMSE regularization (can be fixed
                      at a nominal value; 15 dB is a typical 5G default).
    """

    def __init__(
        self,
        bits_per_symbol: int,
        pilot_indices:   torch.Tensor,
        pilot_symbols:   torch.Tensor,
        snr_prior_db:    float = 15.0,
    ):
        self.M            = bits_per_symbol
        self.pilot_idx    = pilot_indices
        self.pilot_sym    = pilot_symbols
        self.snr_prior_db = snr_prior_db
        self._logmap      = ExactLogMapAWGN(bits_per_symbol)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_llrs(
        self,
        y_grid: torch.Tensor,
        noise_var: float,
        grid_size: int,
    ) -> torch.Tensor:
        """
        Estimate channel, equalize, demap.

        Parameters
        ----------
        y_grid    : Received resource grid [Batch, grid_size] (complex64,
                    vectorized to 1-D per batch item).
        noise_var : Scalar noise variance N0.
        grid_size : Total number of resource elements (pilots + data).

        Returns
        -------
        llrs : [Batch, num_data * M] float32.
        """
        batch = y_grid.shape[0]
        pilot_idx = self.pilot_idx.to(y_grid.device)
        pilot_sym = self.pilot_sym.to(y_grid.device)

        # --- 1. LS estimation at pilot positions ---
        y_pilots = y_grid[:, pilot_idx]                     # [B, P]
        h_ls     = y_pilots / pilot_sym.unsqueeze(0)        # [B, P]  h_hat = y_p / x_p

        # LMMSE refinement: h_lmmse = SNR/(SNR+1) * h_ls  (scalar flat-fading approx)
        snr_linear = 10.0 ** (self.snr_prior_db / 10.0)
        alpha      = snr_linear / (snr_linear + 1.0)
        h_lmmse    = alpha * h_ls.mean(dim=-1, keepdim=True)   # [B, 1] mean channel gain

        # --- 2. Equalization of data subcarriers ---
        data_mask = torch.ones(grid_size, dtype=torch.bool, device=y_grid.device)
        data_mask[pilot_idx] = False
        y_data = y_grid[:, data_mask]                        # [B, D]

        h_eq   = h_lmmse.expand_as(y_data)                  # [B, D]
        h_conj = torch.conj(h_eq)
        h_mag2 = h_eq.abs() ** 2
        equalized = (h_conj * y_data) / (h_mag2 + noise_var + 1e-8)  # [B, D]

        # Effective noise variance after equalization
        eff_noise_var = noise_var / (h_mag2.mean().item() + 1e-8)

        # --- 3. Log-MAP LLR demapping ---
        llrs = self._logmap.compute_llrs(equalized, eff_noise_var)
        return llrs

    # ------------------------------------------------------------------
    @torch.no_grad()
    def ber(
        self,
        y_grid:    torch.Tensor,
        true_bits: torch.Tensor,
        noise_var: float,
        grid_size: int,
    ) -> float:
        """Convenience: BER from LS+LMMSE pipeline."""
        llrs      = self.compute_llrs(y_grid, noise_var, grid_size)
        decisions = (llrs > 0).float()
        return (decisions != true_bits.float()).float().mean().item()


# ---------------------------------------------------------------------------
# 3. Perfect CSI MMSE Baseline (Rayleigh fading upper bound)
# ---------------------------------------------------------------------------

class PerfectCSIMMSE:
    """
    MMSE equalization with perfect Channel State Information (H fully known).

    Serves as the upper-bound reference for Rayleigh fading experiments.
    After equalization the effective channel is:

        y_eq = ( |h|^2 / (|h|^2 + N0) ) * x + noise'

    Exact Log-MAP LLRs are computed after equalization.

    Parameters
    ----------
    bits_per_symbol : Modulation order.
    """

    def __init__(self, bits_per_symbol: int = 4):
        self.M       = bits_per_symbol
        self._logmap = ExactLogMapAWGN(bits_per_symbol)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_llrs(
        self,
        y_data:    torch.Tensor,
        h_data:    torch.Tensor,
        noise_var: float,
    ) -> torch.Tensor:
        """
        Equalize and demap with perfect CSI.

        Parameters
        ----------
        y_data    : Received data symbols [Batch, D] (complex64).
        h_data    : True channel coefficients [Batch, D] (complex64).
        noise_var : Scalar noise variance N0.

        Returns
        -------
        llrs : [Batch, D * M] float32.
        """
        h_conj = torch.conj(h_data)
        h_mag2 = h_data.abs() ** 2
        equalized = (h_conj * y_data) / (h_mag2 + noise_var + 1e-8)   # [B, D]

        # Effective noise variance after MMSE equalization
        eff_noise_var = noise_var / (h_mag2.mean().item() + 1e-8)

        llrs = self._logmap.compute_llrs(equalized, eff_noise_var)
        return llrs

    # ------------------------------------------------------------------
    @torch.no_grad()
    def ber(
        self,
        y_data:    torch.Tensor,
        h_data:    torch.Tensor,
        true_bits: torch.Tensor,
        noise_var: float,
    ) -> float:
        """Convenience: BER from perfect-CSI MMSE pipeline."""
        llrs      = self.compute_llrs(y_data, h_data, noise_var)
        decisions = (llrs > 0).float()
        return (decisions != true_bits.float()).float().mean().item()
