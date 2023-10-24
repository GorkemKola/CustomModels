import torch
from torch import nn



class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def precompute_theta_pos_frequencies(self, head_dim: int, seq_length: int, theta: float = 10000.0):
        assert head_dim % 2 == 0, "number of head dimensions must be divisible by 2"

        # Formula theta_i = 10000**(-2*(i-1)/dim) for i in [1 ... 500]
        theta_numerator = torch.arange(0, head_dim, 2).float()
        
        # as we aranged i [0, 2, 4, ..., head_dim-2], we can say the formula is 10000**(-i/dim) for i in [0, 2, 4, ..., head_dim-2]
        # and it is equal to 1 / 10000**(i/dim)
        theta = 1.0 / (theta ** (theta_numerator / head_dim))
        
        m = torch.arange(seq_length)

        # Shape: (seq_length) outer product (head_dim / 2) --> (seq_lenght, head_dim /2)
        ## Outer product means every element in the first tensor multiplied by evey element in second tensor
        freqs = torch.outer(m, theta).float()

        # We can compute complex numbers as follows:
        # c = R * exp(i*m*theta), where R = 1 => c= exp(i*m*theta)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_complex

    def apply_rottary_embeddings(self, x: torch.Tensor, freqs_complex: torch.Tensor):
        # [x1, x2, x3, ..., xd] -> [[x1, x2], [x3, x4], ..., [xd-1, xd]] -> [x1+i*x2, x3+i*x4, ..., xd-1+i*xd]
        # Shape: (B, seq_length, H, head_dim) -> (B, seq_length, H, head_dim / 2)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # Shape: (seq_length, head_dim / 2) -> (1, seq_length, 1, head_dim / 2)
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

        # cross product of x_complex and freqs_complex
        # Shape: (B, seq_length, H, head_dim / 2) * (1, seq_length, 1, head_dim / 2) --> (B, seq_length, H, head_dim / 3)
        x_rotated = x_complex * freqs_complex

        # Shape: (B, seq_length, H, head_dim / 2) -> (B, seq_length, H, head_dim / 2, 2)
        x_out = torch.view_as_real(x_rotated)

        # Shape: (B, seq_length, H, head_dim / 2, 2) -> (B, seq_length, H, head_dim)
        x_out = x_out.flatten(-2)
        x_out = x_out.type_as(x)
        return x_out
    
    def forward(self, x):
        _, seq_length, _, head_dim = x.shape
        freqs_complex = self.precompute_theta_pos_frequencies(head_dim=head_dim, seq_length=seq_length)
        out = self.apply_rottary_embeddings(x, freqs_complex)
        return out