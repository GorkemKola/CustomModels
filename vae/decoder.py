import torch
from torch import nn
from attention import VAE_AttentionBlock
from residual import VAE_ResidualBlock
class VAE_Decoder(nn.Sequential):

    def __init__(self, 
                 n_channels: int):
        
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),

            nn.Conv2d(128, n_channels, kernel_size=3, padding=1)
        )
        
        # Learnable Scaling Factor
        self.scaling_factor = nn.Parameter(torch.Tensor([0.18215]))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, 4, Height / 8, Width / 8)

        x /= self.scaling_factor

        for module in self:
            x = module(x) 

        return x
