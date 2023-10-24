import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True) -> None:
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, masked=False) -> torch.Tensor:
        # x: (Batch_Size, Seq_Len, Dim)
        batch_size, seq_len, dim = x.shape
        intermim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        # (Batch_Size, Seq_Len, Dim) --> (Batch_Size, Seq_Len, Dim*3) --> 3*(Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_Size, Seq_Len, Dim) --> (Batch_Size, Seq_Len, H, D / H) --> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        
        if masked:
            # Upper Triangle Mask, made with 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            mask = mask * torch.float32("-inf")
            weight += mask
        
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (Batch_size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) --> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, H, Dim / H) --> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(x.shape)

        output = self.out_proj(output)

        return output
    
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channels, Height, Width)

        residue = x
        b, c, h, w = x.shape
        
        # (Batch_Size, Channels, Height, Width) --> (Batch_Size, Channels, Height*Width)
        x = x.view(b, c, h*w)

        # (Batch_Size, Channels, Height*Width) --> (Batch_Size, Height*Width, Channels)
        x = x.transpose(-1, -2)

        # (Batch_Size, Height*Width, Channels) --> (Batch_Size, Height*Width, Channels)
        x = self.attention(x)

        # (Batch_Size, Height*Width, Channels) --> (Batch_Size, Channels, Height*Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Channels, Height*Width) --> (Batch_Size, Channels, Height, Width)
        x = x.view(b, c, h, w)

        return x + residue