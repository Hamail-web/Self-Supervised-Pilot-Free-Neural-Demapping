"""
5G NR OFDM + LDPC Physical Layer Pipeline.

Implements 3GPP Release 17 LDPC encoding/decoding, Gray-mapped QAM constellation mapping,
and 2D OFDM resource grid allocation supporting both:
1. Proposed Pilot-Free Mode (100% data payload, 0% pilot overhead)
2. Classical 3GPP DMRS Mode (14.29% Kronecker pilot overhead on symbols 2 and 11)
"""

import torch
import torch.nn as nn
import numpy as np

try:
    import sionna
    from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
    from sionna.phy.mapping import Mapper, Demapper, Constellation
    from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
    from sionna.phy.utils import ebnodb2no
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False


class OFDMFECPipeline:
    """
    Standard-compliant 5G NR OFDM + LDPC Physical Layer Pipeline.
    """
    def __init__(self,
                 num_ofdm_symbols: int = 14,
                 fft_size: int = 64,
                 subcarrier_spacing: float = 15e3,
                 cyclic_prefix_length: int = 16,
                 num_bits_per_symbol: int = 4,
                 coderate: float = 0.5,
                 num_ldpc_iter: int = 20,
                 use_pilots: bool = False,
                 device: str = "cpu"):
        
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.subcarrier_spacing = subcarrier_spacing
        self.cyclic_prefix_length = cyclic_prefix_length
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate
        self.num_ldpc_iter = num_ldpc_iter
        self.use_pilots = use_pilots
        self.device = torch.device(device)
        
        # Total resource elements in slot
        self.total_res = num_ofdm_symbols * fft_size  # 14 * 64 = 896
        
        if use_pilots:
            # Standard 3GPP 2-symbol pilot pattern (symbols 2 and 11)
            # Pilot overhead: 2 / 14 = 14.29%
            self.pilot_symbols = [2, 11]
            self.pilot_mask = torch.zeros((fft_size, num_ofdm_symbols), dtype=torch.bool)
            for sym in self.pilot_symbols:
                self.pilot_mask[:, sym] = True
            self.num_pilot_res = self.pilot_mask.sum().item()
            self.num_data_res = self.total_res - self.num_pilot_res
        else:
            # 100% payload utilization (0% pilot overhead)
            self.pilot_symbols = []
            self.pilot_mask = torch.zeros((fft_size, num_ofdm_symbols), dtype=torch.bool)
            self.num_pilot_res = 0
            self.num_data_res = self.total_res
            
        self.n = self.num_data_res * self.num_bits_per_symbol
        self.k = int(self.n * self.coderate)
        
        if SIONNA_AVAILABLE:
            # Sionna-based 5G NR LDPC encoder and BP decoder
            self.encoder = LDPC5GEncoder(k=self.k, n=self.n).to(self.device)
            self.decoder = LDPC5GDecoder(self.encoder, num_iter=self.num_ldpc_iter).to(self.device)
            self.constellation = Constellation("qam", num_bits_per_symbol=self.num_bits_per_symbol).to(self.device)
            self.mapper = Mapper(constellation=self.constellation).to(self.device)
            self.classical_demapper = Demapper("app", constellation=self.constellation).to(self.device)
        else:
            self.encoder = None
            self.decoder = None

    def generate_random_bits(self, batch_size: int):
        """Generates random information bits of shape [Batch, k]."""
        return torch.randint(0, 2, (batch_size, self.k), dtype=torch.float32, device=self.device)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """Encodes information bits [Batch, k] into codeword [Batch, n]."""
        if SIONNA_AVAILABLE and self.encoder is not None:
            return self.encoder(u)
        else:
            # Fallback systematic repetition for environment without Sionna
            repeats = self.n // self.k
            c = u.repeat(1, repeats)
            if c.shape[-1] < self.n:
                c = torch.cat([c, u[:, :self.n - c.shape[-1]]], dim=-1)
            return c

    def decode(self, llrs: torch.Tensor):
        """
        Runs Belief Propagation decoding on soft LLRs.
        Returns:
            u_hat : Decoded information bits [Batch, k]
            c_hat : Decoded codeword bits [Batch, n]
        """
        if SIONNA_AVAILABLE and self.decoder is not None:
            u_hat = self.decoder(llrs)
            # Re-encode to obtain full decoded codeword for syndrome check
            c_hat = self.encoder(u_hat)
            return u_hat, c_hat
        else:
            # Fallback hard decision slicing
            hard_bits = (llrs > 0).float()
            u_hat = hard_bits[:, :self.k]
            c_hat = hard_bits
            return u_hat, c_hat

    def map_to_grid(self, c: torch.Tensor) -> torch.Tensor:
        """
        Maps coded bits [Batch, n] into constellation symbols and places them into the 2D grid.
        Returns:
            grid : Complex tensor of shape [Batch, fft_size, num_ofdm_symbols]
        """
        batch_size = c.shape[0]
        if SIONNA_AVAILABLE and self.mapper is not None:
            symbols = self.mapper(c)  # [Batch, num_data_res]
        else:
            # Basic QAM mapping fallback
            symbols = (2.0 * c[:, ::2] - 1.0) + 1j * (2.0 * c[:, 1::2] - 1.0)
            symbols = symbols / np.sqrt(2.0)
            
        grid = torch.zeros((batch_size, self.fft_size, self.num_ofdm_symbols), dtype=torch.complex64, device=self.device)
        
        if self.use_pilots:
            # Place known pilot symbol (1+1j)/sqrt(2) at pilot REs
            pilot_val = (1.0 + 1.0j) / np.sqrt(2.0)
            grid[:, self.pilot_mask] = pilot_val
            grid[:, ~self.pilot_mask] = symbols
        else:
            # 100% payload
            grid = symbols.view(batch_size, self.fft_size, self.num_ofdm_symbols)
            
        return grid
