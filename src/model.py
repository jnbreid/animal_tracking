
import torch
from torch import nn

class DistNet_t(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin_relu = nn.Sequential(
        nn.Linear(3,1),
    )
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = torch.reshape(x, (-1,3))
    x = self.lin_relu(x)
    x = self.sigmoid(x)
    return x

class DistNet_m(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin_relu = nn.Sequential(
        nn.Linear(3,5),
        nn.LeakyReLU(),
        nn.Linear(5,5),
        nn.LeakyReLU(),
        nn.Linear(5,1),
    )
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = torch.reshape(x, (-1,3))
    x = self.lin_relu(x)
    x = self.sigmoid(x)
    return x
  
class DistNet_l(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin_relu = nn.Sequential(
        nn.Linear(3,5),
        nn.LeakyReLU(),
        nn.Linear(5,5),
        nn.LeakyReLU(),
        nn.Linear(5,1),
    )
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = torch.reshape(x, (-1,3))
    x = self.lin_relu(x)
    x = self.sigmoid(x)
    return x
    