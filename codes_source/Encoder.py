import torch
import torch.nn as nn

class Encoder(nn.Module):
   
    def __init__(self, in_channels=3, z_channels=512):
        super().__init__()

        self.net = nn.Sequential(
            # 256x256 -> 128x128, C16
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128 -> 64x64, C32
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32, C64
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16, C128
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8, C256
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4, C512
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 2x2, C512 (z)
            nn.Conv2d(512, z_channels, 4, 2, 1),
            nn.BatchNorm2d(z_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        z = self.net(x)
        return z