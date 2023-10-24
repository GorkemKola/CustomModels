import torch
from torch import nn
from encoder import VAE_Encoder
from decoder import VAE_Decoder

class VAE(nn.Module):

    def __init__(self, 
                 n_channels: int) -> None:
        
        super().__init__()

        self.encoder = VAE_Encoder(n_channels)
        self.decoder = VAE_Decoder(n_channels)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)

        # (Batch_Size, Channel, Height, Width) --> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.encoder(x, noise)

        # (Batch_Size, 4, Height / 8, Width / 8) --> (Batch_Size, Channel, Height, Width)
        x = self.decoder(x)

        return x
    
