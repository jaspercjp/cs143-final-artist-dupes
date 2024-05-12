import torch.nn as nn
import torch
from torchinfo import summary

# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation="silu"):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        if activation=="silu":
            self.activation = nn.SiLU()
        elif activation==None:
            self.activation = lambda x: x
        self.norm = nn.BatchNorm2d(out_channels)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) #, padding)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        x = self.norm(x)
        return self.activation(x)

# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        self.reflection_pad = nn.ReflectionPad2d(kernel_size//2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        x = self.norm(x)
        return self.silu(x)

# Residual Block
#   adapted from pytorch tutorial
#   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-
#   intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, activation=None)

    def forward(self, x):
        y = self.conv1(x)
        return self.conv2(y) + x

# Adapted from https://github.com/dxyang/StyleTransfer/blob/master/style.py
class ImageTransformerRef(nn.Module):
    def __init__(self):
        super(ImageTransformerRef, self).__init__()

        # encoding layers
        self.in_layer = ConvLayer(3, 32, 9, 1)
        self.down1 = ConvLayer(32, 64, 3, 2)
        self.down2 = ConvLayer(64, 128, 3, 2)

        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        self.up2 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.up1 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.out_layer = ConvLayer(32, 3, 9, 1, activation=None)

    def forward(self, x):
        # encode
        x = self.in_layer(x)
        x = self.down1(x)
        x = self.down2(x)

        # residual layers
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # decode
        x = self.up2(x)
        x = self.up1(x)
        x = self.out_layer(x)

        return x

if __name__ == "__main__":
    model = ImageTransformerRef()
    summary(model, (5, 3, 256, 256))
        
