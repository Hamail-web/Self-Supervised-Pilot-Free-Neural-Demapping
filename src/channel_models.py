"""
Channel Models for Self-Supervised Pilot-Free Neural Demapping.

Provides pure-PyTorch (Sionna-independent) channel simulation:
  - AWGNChannel       : Additive White Gaussian Noise
  - RayleighFadingChannel : Block-fading Rayleigh channel with optional Doppler
  - DynamicChannelScheduler : Switches channel type mid-experiment for tracking studies

All classes operate on complex64 tensors shaped [Batch, Subcarriers, Symbols].
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# AWGN Channel
# ---------------------------------------------------------------------------

class AWGNChannel:
    """
    Additive White Gaussian Noise channel.

    The channel model is:  y = x + n,  n ~ CN(0, sigma^2 I)

    where sigma^2 = 1 / (2 * SNR_linear) for normalised unit-power symbols.
    """

    def __call__(
        self,
        x: torch.Tensor,
        snr_db: float,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : Complex tensor [Batch, Subcarriers, Symbols] (unit-power symbols).
        snr_db  : Signal-to-Noise Ratio in dB.

        Returns
        -------
        y : Noisy received signal, same shape as x.
        """
        sigma2 = 10.0 ** (-snr_db / 10.0)          # noise power per dimension pair
        sigma = (sigma2 / 2.0) ** 0.5               # std per real/imag component
        noise_re = torch.randn_like(x.real) * sigma
        noise_im = torch.randn_like(x.imag) * sigma
        noise = torch.complex(noise_re, noise_im)
        return x + noise

    def get_noise_var(self, snr_db: float) -> float:
        """Return total complex noise variance (sigma^2 = N0) for given SNR in dB."""
        return 10.0 ** (-snr_db / 10.0)


# ---------------------------------------------------------------------------
# Rayleigh Block-Fading Channel
# ---------------------------------------------------------------------------

class RayleighFadingChannel:
    """
    Frequency-flat block Rayleigh fading channel with optional Doppler phase rotation.

    The channel model per slot t is:

        y[t] = h[t] * x[t] + n[t]

    where
        h[t] ~ CN(0, 1)   (block-fading: constant within one OFDM symbol,
                           correlated across symbols via first-order AR model when
                           doppler_hz > 0)
        n[t] ~ CN(0, sigma^2 I)

    Parameters
    ----------
    doppler_hz       : Maximum Doppler frequency in Hz (0 = quasi-static block fading).
    sample_rate_hz   : OFDM symbol rate (number of OFDM symbols per second). Defaults
                       to a typical 5G NR value of ~14e3 (14 symbols per 1 ms slot).
    """

    def __init__(self, doppler_hz: float = 0.0, sample_rate_hz: float = 14000.0):
        self.doppler_hz = doppler_hz
        self.sample_rate_hz = sample_rate_hz
        # Jakes' first-order AR correlation coefficient between adjacent OFDM symbols
        self._rho = float(np.j0(2 * np.pi * doppler_hz / sample_rate_hz)) if doppler_hz > 0 else 1.0

    def __call__(
        self,
        x: torch.Tensor,
        snr_db: float,
        return_channel: bool = False,
    ):
        """
        Parameters
        ----------
        x              : Complex tensor [Batch, Subcarriers, Symbols].
        snr_db         : Signal-to-Noise Ratio in dB.
        return_channel : If True, also return the true channel matrix H.

        Returns
        -------
        y          : Received signal [Batch, Subcarriers, Symbols].
        H (optional): True channel coefficients [Batch, Subcarriers, Symbols].
        """
        batch, n_sc, n_sym = x.shape
        device = x.device

        sigma2 = 10.0 ** (-snr_db / 10.0)
        sigma  = (sigma2 / 2.0) ** 0.5

        # --- Generate correlated channel taps across OFDM symbols ---
        rho = self._rho
        H = torch.zeros(batch, n_sc, n_sym, dtype=torch.complex64, device=device)

        # First symbol: i.i.d. CN(0,1)
        h_re = torch.randn(batch, n_sc, device=device) / (2.0 ** 0.5)
        h_im = torch.randn(batch, n_sc, device=device) / (2.0 ** 0.5)
        H[:, :, 0] = torch.complex(h_re, h_im)

        innov_std = ((1.0 - rho ** 2) / 2.0) ** 0.5  # innovation std per dim
        for t in range(1, n_sym):
            innov_re = torch.randn(batch, n_sc, device=device) * innov_std
            innov_im = torch.randn(batch, n_sc, device=device) * innov_std
            H[:, :, t] = rho * H[:, :, t - 1] + torch.complex(innov_re, innov_im)

        # --- Apply fading and add noise ---
        y = H * x
        noise = torch.complex(
            torch.randn_like(x.real) * sigma,
            torch.randn_like(x.imag) * sigma,
        )
        y = y + noise

        if return_channel:
            return y, H
        return y

    def get_noise_var(self, snr_db: float) -> float:
        """Return total complex noise variance N0 for given SNR in dB."""
        return 10.0 ** (-snr_db / 10.0)


# ---------------------------------------------------------------------------
# Dynamic Channel Scheduler (for Experiment 04)
# ---------------------------------------------------------------------------

class DynamicChannelScheduler:
    """
    Schedules a sequence of channel configurations to simulate changing conditions.

    Used in Experiment 04 to demonstrate that the online self-supervised adaptation
    can recover BER after a sudden channel transition (AWGN → Rayleigh + Doppler).

    Parameters
    ----------
    schedule : list of dict
        Each dict has keys:
          - 'start_slot' (int)   : OFDM slot index from which this config activates.
          - 'channel'    (str)   : 'awgn' | 'rayleigh'.
          - 'snr_db'     (float) : SNR for this phase.
          - 'doppler_hz' (float, optional): Doppler frequency (Rayleigh only).
    """

    def __init__(self, schedule: list):
        # Sort by start_slot ascending
        self.schedule = sorted(schedule, key=lambda s: s['start_slot'])
        self._awgn = AWGNChannel()

    def _get_channel(self, slot: int):
        """Return the active (channel_instance, snr_db) for a given slot index."""
        active = self.schedule[0]
        for entry in self.schedule:
            if slot >= entry['start_slot']:
                active = entry
            else:
                break

        if active['channel'] == 'awgn':
            return self._awgn, active['snr_db']
        elif active['channel'] == 'rayleigh':
            ch = RayleighFadingChannel(
                doppler_hz=active.get('doppler_hz', 0.0),
                sample_rate_hz=active.get('sample_rate_hz', 14000.0),
            )
            return ch, active['snr_db']
        else:
            raise ValueError(f"Unknown channel type: {active['channel']}")

    def transmit(self, x: torch.Tensor, slot: int, return_channel: bool = False):
        """
        Apply the scheduled channel at the given slot index.

        Parameters
        ----------
        x            : Transmit signal [Batch, Subcarriers, Symbols].
        slot         : Current slot (time) index.
        return_channel: Pass through to Rayleigh channel.

        Returns
        -------
        y          : Received signal.
        channel_id : String label of the active channel for logging.
        """
        ch, snr_db = self._get_channel(slot)
        label = type(ch).__name__

        if isinstance(ch, AWGNChannel):
            y = ch(x, snr_db)
            if return_channel:
                # Perfect channel for AWGN is H = 1
                H = torch.ones_like(x)
                return y, H, label
            return y, label

        elif isinstance(ch, RayleighFadingChannel):
            if return_channel:
                y, H = ch(x, snr_db, return_channel=True)
                return y, H, label
            y = ch(x, snr_db, return_channel=False)
            return y, label

    def get_schedule_summary(self) -> str:
        """Return a human-readable summary of the channel schedule."""
        lines = ["DynamicChannelScheduler schedule:"]
        for entry in self.schedule:
            d = entry.get('doppler_hz', 0.0)
            lines.append(
                f"  Slot {entry['start_slot']:4d}+  |  {entry['channel'].upper():8s} "
                f"|  SNR={entry['snr_db']:5.1f} dB  |  Doppler={d:.1f} Hz"
            )
        return "\n".join(lines)
