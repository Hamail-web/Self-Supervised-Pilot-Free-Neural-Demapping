"""
Neural Demapper Architecture for 2D OFDM Resource Grids.

Processes raw time-frequency OFDM resource grids [Batch, 2, Subcarriers, Symbols]
and outputs calibrated soft Log-Likelihood Ratios (LLRs) for downstream 5G NR LDPC decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    """2D Convolutional Block with Group Normalization and GELU activation."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock2D(nn.Module):
    """2D Residual Convolutional Block to preserve fine time-frequency channel correlations."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + residual)


class ResCNNDemapper(nn.Module):
    """
    Joint 2D Time-Frequency Residual CNN Neural Demapper.

    Parameters:
    -----------
    num_bits_per_symbol : int
        Modulation order (2 for QPSK, 4 for 16-QAM, 6 for 64-QAM).
    hidden_dim : int
        Number of hidden feature channels (default: 48 for 16-QAM, 64 for 64-QAM).
    num_res_blocks : int
        Number of residual convolutional blocks.
    temperature : float
        Initial learnable temperature scaling parameter for LLR calibration.
    """
    def __init__(self,
                 num_bits_per_symbol: int = 4,
                 hidden_dim: int = 64,
                 num_res_blocks: int = 3,
                 temperature: float = 1.0):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.hidden_dim = hidden_dim
        
        # Ingest Real and Imaginary components (2 input channels)
        self.input_proj = ConvBlock2D(in_channels=2, out_channels=hidden_dim)
        
        # Residual feature extraction backbone across subcarriers and OFDM symbols
        self.res_blocks = nn.ModuleList([
            ResBlock2D(channels=hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # Projection head producing soft logit estimates per constellation bit
        self.output_conv = nn.Conv2d(hidden_dim, num_bits_per_symbol, kernel_size=3, padding=1, bias=True)
        
        # Learnable temperature scaling for calibrated soft LLRs
        self.scale = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, y_grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
        -----------
        y_grid : torch.Tensor
            Complex tensor of shape [Batch, Subcarriers, Symbols]
            or real-valued tensor of shape [Batch, 2, Subcarriers, Symbols].

        Returns:
        --------
        llrs : torch.Tensor
            Calibrated soft LLRs of shape [Batch, Total_Coded_Bits].
        """
        if torch.is_complex(y_grid):
            x = torch.stack([y_grid.real, y_grid.imag], dim=1).float()
        else:
            x = y_grid

        feat = self.input_proj(x)
        for block in self.res_blocks:
            feat = block(feat)

        # Shape: [Batch, num_bits_per_symbol, Subcarriers, Symbols]
        raw_logits = self.output_conv(feat)
        calibrated_llrs = raw_logits * self.scale

        # Reshape to [Batch, Subcarriers * Symbols * num_bits_per_symbol]
        b, m, sc, sym = calibrated_llrs.shape
        llrs = calibrated_llrs.permute(0, 2, 3, 1).contiguous().view(b, -1)
        return llrs

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
