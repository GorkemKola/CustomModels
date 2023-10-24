import torch
from torch import nn

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channels, Height, Width)
        residue = x
        x = self.groupnorm_1(x)
        x = nn.SiLU()(x)
        x = self.conv1(x)

        x = self.groupnorm_2(x)
        x = nn.SiLU()(x)
        x = self.conv2(x)

        x = self.residual_layer(residue) + x
        return x
    