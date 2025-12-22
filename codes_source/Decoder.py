import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, n_attr, z_channels=512):
        super().__init__()
        self.n_attr = n_attr

        # on concatène z (512 canaux) et y (n_attr canaux) à l'entrée
        in_channels = z_channels + n_attr

        self.net = nn.Sequential(
            # 2x2 -> 4x4, C512
            nn.ConvTranspose2d(in_channels, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4x4 -> 8x8, C512
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8x8 -> 16x16, C256
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16x16 -> 32x32, C128
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32x32 -> 64x64, C64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64x64 -> 128x128, C32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 128x128 -> 256x256, C16
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # sortie : 3 canaux RGB
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),   # valeurs dans [-1, 1]
        )

    def forward(self, z, y):
        """
        z : [B, 512, 2, 2]
        y : [B, n_attr]
        """
        B, C, H, W = z.shape

        # étaler y sur HxW pour concaténer avec z
        y_exp = y.view(B, self.n_attr, 1, 1).expand(B, self.n_attr, H, W)
        h = torch.cat([z, y_exp], dim=1)    

        x_hat = self.net(h)                 
        return x_hat