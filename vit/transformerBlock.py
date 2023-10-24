import torch
import torch.nn as nn
from .mlp import MLP
from .attention import Attention
class Block(nn.Module):
    """
        Transformer Block

        Parameters
        ______________

        dim : int
            Embedding dimension
        
        n_heads : int
            number of attention heads

        mlp_ratio : float
            Determines the hidden dimension size of the \\ 
            `MLP` module with respect to `dim`

        qkv_bias : bool
            If true then we include bias to quer, key, value projections.

        p, attn_p : float
            Dropout probablity

        Attributes
        _______________

        norm1, norm2 : LayerNorm
            Layer Normalization

        attn : Attention
            Attention Module
        
        mlp : MLP
            MLP Module
    """
    def __init__(self, 
                 dim: int, 
                 n_heads: int, 
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 p: float = .0,
                 attn_p: float = .0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, 1e-6)

        self.attn = Attention(
            dim,
            n_heads=n_heads, 
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        hidden_features = int(dim * mlp_ratio)

        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
            p=p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_patches + 1, dim)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x