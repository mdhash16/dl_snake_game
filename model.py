import torch
import torch.nn as nn


class LinearQNet(nn.Module):
    """
    Simple feed-forward Q-network.

    Architecture: 11 inputs → 256 hidden (ReLU) → 3 outputs

    Input  (11): the game state from InputLayer.get_state()
    Output  (3): Q-values for [go straight, turn right, turn left]
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.net(x)

    def save(self, path='model.pth'):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path='model.pth'):
        model = cls()
        model.load_state_dict(torch.load(path, weights_only=True))
        return model
