"""
adapted from `disent/model/ae/_vae_conv64.py`, which is Copyright (c) 2021 Nathan Juraj Michlo, with MIT License
"""

from disent.model import DisentDecoder, DisentEncoder
from torch import Tensor, nn


class EncoderConv28(DisentEncoder):
    def __init__(self, x_shape=(3, 28, 28), z_size=6, z_multiplier=1):
        (C, H, W) = x_shape
        assert (H, W) == (28, 28), "This model only works with image size 28x28."
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            nn.Conv2d(C, 32, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 7),  # 1x1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.z_total),
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


class DecoderConv28(DisentDecoder):
    def __init__(self, x_shape=(3, 28, 28), z_size=6, z_multiplier=1):
        (C, H, W) = x_shape
        assert (H, W) == (28, 28), "This model only works with image size 28x28."
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            nn.Linear(self.z_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Unflatten(1, (128, 1, 1)),
            nn.ConvTranspose2d(128, 64, 7),  # 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, C, 4, stride=2, padding=1),  # 28x28
        )

    def decode(self, z) -> Tensor:
        return self.model(z)
