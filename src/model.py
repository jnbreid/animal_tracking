
import torch
from torch import nn

class DistNet_t(nn.Module):
    """
    Model class for DistNet (tiny-sized version).

    Architecture:
        Linear(3 → 1) → Sigmoid

    A simple neural network that maps a 3-dimensional input vector to a scalar output.

    Input:
        - torch.Tensor of shape [3] (or [batch_size, 3])
    Output:
        - torch.Tensor of shape [1] (or [batch_size, 1])
    """
    def __init__(self):
        """
        Initializes the DistNet_t model with a single linear layer followed by a sigmoid activation.
        """
        super().__init__()
        self.lin_relu = nn.Sequential(
            nn.Linear(3, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.reshape(x, (-1, 3))
        x = self.lin_relu(x)
        x = self.sigmoid(x)
        return x

class DistNet_m(nn.Module):
    """
    Model class for DistNet (medium-sized version).

    Architecture:
        Linear(3 → 5) → LeakyReLU →
        Linear(5 → 5) → LeakyReLU →
        Linear(5 → 1) → Sigmoid


    Input:
        - torch.Tensor of shape [3] (or [batch_size, 3])
    Output:
        - torch.Tensor of shape [1] (or [batch_size, 1])
    """
    def __init__(self):
        """
        Initializes the DistNet_m model with two hidden layers and LeakyReLU activations,
        followed by a final linear layer and sigmoid activation.
        """
        super().__init__()
        self.lin_relu = nn.Sequential(
            nn.Linear(3, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.reshape(x, (-1, 3))
        x = self.lin_relu(x)
        x = self.sigmoid(x)
        return x
  
class DistNet_l(nn.Module):
    """
    Model class for DistNet (large version).

    Architecture:
        Linear(3 → 10) → LeakyReLU →
        Linear(10 → 10) → LeakyReLU →
        Linear(10 → 10) → LeakyReLU →
        Linear(10 → 1) → Sigmoid


    Input:
        - torch.Tensor of shape [3] (or [batch_size, 3])
    Output:
        - torch.Tensor of shape [1] (or [batch_size, 1])
    """
    def __init__(self):
        """
        Initializes the DistNet_l model with three hidden layers of size 10 and LeakyReLU activations,
        followed by a final linear layer and a sigmoid activation.
        """
        super().__init__()
        self.lin_relu = nn.Sequential(
            nn.Linear(3, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.reshape(x, (-1, 3))
        x = self.lin_relu(x)
        x = self.sigmoid(x)
        return x
    