import torch.nn as nn


## Layers used in VAE
class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


def get_architectures(hidden_dims: int):
    return {
        "conv3": {
            "encoder": nn.Sequential(
                nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 14x14
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 7x7
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 7),  # 1x1
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128, hidden_dims),
                nn.ReLU(),
            ),
            "decoder": nn.Sequential(
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, 128),
                nn.ReLU(),
                Reshape((-1, 128, 1, 1)),
                nn.ConvTranspose2d(128, 64, 7),  # 7x7
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 28x28
                nn.Sigmoid(),
            ),
            "encode_dim": hidden_dims,
            "decode_dim": hidden_dims,
        },
        "simple64bn": {
            "encoder": nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 32x32
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 8x8
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 4x4
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, hidden_dims),
                nn.ReLU(),
            ),
            "decoder": nn.Sequential(
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, 256 * 4 * 4),
                nn.ReLU(),
                Reshape((-1, 256, 4, 4)),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 64x64
                nn.Sigmoid(),
            ),
            "encode_dim": hidden_dims,
            "decode_dim": hidden_dims,
        },
    }
