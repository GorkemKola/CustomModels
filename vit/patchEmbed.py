import torch
from torch import nn
from torch.nn import functional as F
class PatchEmbed(nn.Module):
    """
        Split image  into patches and then embed them.

        Parameters
        ______________

        img_size : int
            Size of the image (NxN)
        
        path_size : int
            Size of the patch (NxN)

        in_chans : int
            Number of input channels
        
        embed_dim : int
            The embedding Dimension
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size)**2 + ((img_size - patch_size) // patch_size)**2
        
        self.proj = nn.Conv2d(in_channels=in_chans, 
                              out_channels=embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, img_size, img_Size)
        pad = - self.patch_size // 2
        # (batch_size, channels, img_size - patch_size, img_size - patch_size)
        padded_x = F.pad(x, (pad, pad, pad, pad))
        # (batch_size, channels, img_size, img_Size) --> (batch_size, embed_dim, n_patches**0.5, n_patches**0.5)
        x = self.proj(x)
        padded_x = self.proj(padded_x)


        # (batch_size, channels, n_patches**0.5, n_patches**0.5) --> (batch_size, embed_dim, n_patches)
        x = torch.cat([x.flatten(2), padded_x.flatten(2)], dim=2)
        # (batch_size, embed_dim, n_patches) --> (batch_size, n_patches, embed_dim)
        x = x.transpose(1, 2)
        print(x.shape)
        return x