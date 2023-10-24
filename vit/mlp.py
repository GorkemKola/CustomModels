import torch
import torch.nn as nn

class MLP(nn.Module):
    """
        Multilayer Perceptron

        Parameters
        ______________

        in_features : int
            Number of input features

        hidden_features : int
            Number of nodes in hidden layer

        out_features : int
            Number of output features

        p : float
            dropout probablity

        Attributes
        _______________

        fc1 : nn.Linear
            the first linear layer

        act : nn.GELU
            GELU activation function

        fc2 : nn.Lİnear
            the second linear layer
        
        drop : nn.Dropout
            Dropout layer
    """
    def __init__(self, 
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 p: float) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, n_patches + 1, in_features)

        x = self.fc1(x) # (batch_size, n_patches + 1, hidden_features)
        
        x = self.act(x)

        x = self.drop(x)

        x = self.fc2(x) # (batch_size, n_patches + 1, out_features)

        x = self.act(x)

        return x