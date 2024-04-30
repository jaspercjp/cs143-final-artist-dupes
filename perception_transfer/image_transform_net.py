import torch.nn as nn
from residual_block import ResidualBlock

class ImageTransformer(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 81, image_size = (256, 256)):
        super().__init__()

        # two stride-2 convolutions to downsample
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels, 
                kernel_size=3, 
                stride=2,
                padding=1
            ), 
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels, 
                kernel_size=3, 
                stride=2,
                padding=1
            ), 
            nn.LeakyReLU(),

        )

        # 3 residual blocks?
        self.residuals = nn.Sequential(
            ResidualBlock(in_channels= out_channels, out_channels=out_channels),
            ResidualBlock(in_channels= out_channels, out_channels=out_channels),
            ResidualBlock(in_channels= out_channels, out_channels=out_channels),
        )

        # upscale block
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = out_channels,
                out_channels = out_channels, 
                kernel_size=3, 
                stride=2,
                padding=1, 
                output_padding=1
            ), 
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels = out_channels,
                out_channels = in_channels, 
                kernel_size=3, 
                stride=2,
                padding=1, 
                output_padding=1
            ), 
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.down(x)
        x = self.residuals(x)
        x = self.up(x)
        return x