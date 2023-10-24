import torch
import torch.nn as nn
from .patchEmbed import PatchEmbed
from .transformerBlock import Block

class ViT(nn.Module):
    """
        Simplified Implementation of the Vision Transformer.

        Parameters
        ______________

        img_size : int
            Both height and width of the image.

        patch_size : int
            Both height and width of the patch

        in_chans : int
            Number of input channels

        n_classes : int
            Number of classes

        embed_dim : int
            Dimension of embeddings

        depth : int
            Number of blocks
        
        n_heads : int
            Number of attention heads

        mlp_ratio : float
            Determines the hidden dimension of the `MLP` module.

        qkv_bias : bool
            If true then we include bias to the query, key and value projections.
        
        p, attn_p : float
            Dropout probablity

        Attributes
        _______________

        patch_embed : PatchEmbed
            Instance of `PatchEmbed` layer
        
        cls_token : nn.Parameter
            Learnable parameter that will represent the first token in the sequence.
            Ut has `embed_dim` elements

        pos_emb : nn.Parameter
            Positional embeddings of the cls token + all the patches.
            It has `(n_patches + 1) * embed_dim` elements

        pos_drop : nn.Dropout
            Dropout Layer.

        norm: nn.LayerNorm
            Layer Normalization
    """

    def __init__(self, 
                 img_size: int = 384,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 n_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 n_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 p: float = .0,
                 attn_p: float = .0) -> None:
        super().__init__()

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )

        self.pos_drop = nn.Dropout(p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, img_size, img_size)
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        ) # (n_samples, 1, dim)

        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits

