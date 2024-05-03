import torch.nn as nn
from residual_block import ResidualBlock

class ImageTransformer(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 128, image_size = (256, 256)):
        super().__init__()

        # two stride-2 convolutions to downsample
        self.down = nn.Sequential(
            # first conv layer decreases the aliasing
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = 32, 
                kernel_size=9, 
                stride=1,
                padding=4, 
                # padding_mode='reflect'
            ), 
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # second conv layer downsamples
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64, 
                kernel_size=3, 
                stride=2,
                padding=1, 
                # padding_mode='reflect'
            ), 
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # third conv layer downsamples
            nn.Conv2d(
                in_channels = 64,
                out_channels = 128, 
                kernel_size=3, 
                stride=2,
                padding=1, 
                # padding_mode='reflect'
            ), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )

        # 7 residual blocks?
        self.residuals = nn.Sequential(
            ResidualBlock(in_channels= out_channels, out_channels=out_channels),
            ResidualBlock(in_channels= out_channels, out_channels=out_channels),
            ResidualBlock(in_channels= out_channels, out_channels=out_channels),
            ResidualBlock(in_channels= out_channels, out_channels=out_channels),
            ResidualBlock(in_channels= out_channels, out_channels=out_channels),
            ResidualBlock(in_channels= out_channels, out_channels=out_channels),
            ResidualBlock(in_channels= out_channels, out_channels=out_channels),
        )

        # upscale block
        self.up = nn.Sequential(
            
            # upsample 1
            nn.ConvTranspose2d(
                in_channels = 128,
                out_channels = 64, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                output_padding=1
            ), 
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # upsample 2
            nn.ConvTranspose2d(
                in_channels = 64,
                out_channels = 32, 
                kernel_size=3, 
                stride=2,
                padding=1, 
                output_padding=1
            ), 
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # smooth
            nn.Conv2d(
                in_channels = 32,
                out_channels = in_channels, 
                kernel_size=9, 
                stride=1,
                padding=4, 
                # padding_mode='reflect'
            ), 
            # nn.BatchNorm2d(in_channels),
            # nn.Sigmoid(),


        )

    def forward(self, x):
        x = self.down(x)
        x = self.residuals(x)
        x = self.up(x)
        return x