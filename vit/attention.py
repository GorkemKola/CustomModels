import torch
import torch.nn as nn
from .rotaryEmbeddings import RotaryPositionalEmbeddings

class Attention(nn.Module):
    
    """
        Attention Mechanism

        Parameters
        ______________
        
        dim : int
            The input and output dimension of per token features.

        n_heads : int
            Number of attention heads.

        qkv_bias : bool
            If true then we include bias to the query, key and value projections.

        attn_p : float
            Dropout probablity applied after softmax operation.

        proj_p : float
            Dropout probablity applied to the output tensor.

        Attributes
        _______________

        scale : float
            Normalizing constant for the dot product.

        qkv : nn.Linear
            Linear projection for the query, key and value

        proj : nn.Linear
            Linear mapping that takes in the concatanated \\
            output of all attention heads, and maps it into a new space

        attn_drop, proj_drop : nn.Dropout
            Dropout layers
        
        Formula
        ________________
        
        dim => d
        q, k, v => Q, K, V
        scale => 1 / sqrt(d)

        Attention = softmax(Q*K.T/sqrt(d)) * V
        
    """

    def __init__(self, dim: int, 
                 n_heads: int = 12, 
                 qkv_bias: bool = True, 
                 attn_p: float = .0,
                 proj_p: float = .0) -> None:
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -.5
        self.pe = RotaryPositionalEmbeddings()
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_patches + 1, dim)
        
        batch_size, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) # (batch_size, n_patches + 1, 3*dim)

        qkv = qkv.reshape(
            batch_size, n_tokens, 3, self.n_heads, self.head_dim
        ) # (batch_size, n_patches + 1, 3, heads, head_dim)

        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ) # (3, batch_size, heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Positional Encoding to Query and Key Matrices like as in LLAMA.
        q, k = self.pe(q), self.pe(k)

        k_t = k.transpose(-2, -1) # (batch_size, heads, head_dim, n_patches + 1) 

        dp = (q @ k_t) * self.scale # (batch_size, heads, n_patches + 1, n_patches + 1)

        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v # (batch_size, heads, n_patches + 1, head_dim)
        
        weighted_avg = weighted_avg.transpose(
            1, 2
        ) # (batch_size, n_patches + 1, heads, head_dim)

        weighted_avg = weighted_avg.reshape(
            batch_size, n_tokens, dim
        )

        x = self.proj(weighted_avg) # (batch_size, n_patches + 1, dim)
        x = self.proj_drop(x)

        return x