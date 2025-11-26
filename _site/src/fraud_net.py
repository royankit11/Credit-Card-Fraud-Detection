import torch
import torch.nn as nn

class FraudNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(128, 64)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
