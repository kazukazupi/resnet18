import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, en_downsample: bool = False
    ):
        """
        Args:
            in_channels: 入力のチャネル数
            out_channels: 出力のチャネル数
            en_downsample: Trueなら入力サイズを半分にし、Falseなら維持する
        """

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2 if en_downsample else 1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            if en_downsample
            else None
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            y += self.bn3(self.downsample(x))
        else:
            y += x

        y = self.relu2(y)

        return y


class Resnet18(nn.Module):

    def __init__(self, en_rgb: bool = True, num_classes: int = 10):

        super(Resnet18, self).__init__()

        # conv1
        self.conv1 = nn.Conv2d(
            3 if en_rgb else 1, 64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # conv2_x
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(64, 64, False)
        self.res_block2 = ResidualBlock(64, 64, False)

        # conv3_x
        self.res_block3 = ResidualBlock(64, 128, True)
        self.res_block4 = ResidualBlock(128, 128, False)

        # conv4_x
        self.res_block5 = ResidualBlock(128, 256, True)
        self.res_block6 = ResidualBlock(256, 256, False)

        # conv5_x
        self.res_block7 = ResidualBlock(256, 512, True)
        self.res_block8 = ResidualBlock(512, 512, False)

        # average pool
        self.avg_pool = nn.AvgPool2d(7)

        # classifier
        self.linear = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:

        # conv1
        feature = self.conv1(x)  # (batch_size, 64, 112, 112)

        # conv2_x
        feature = self.max_pool1(feature)  # (batch_size, 64, 56, 56)
        feature = self.res_block1(feature)  # (batch_size, 64, 56, 56)
        feature = self.res_block2(feature)  # (batch_size, 64, 56, 56)

        # conv3_x
        feature = self.res_block3(feature)  # (batch_size, 128, 28, 28)
        feature = self.res_block4(feature)  # (batch_size, 128, 28, 28)

        # conv4_x
        feature = self.res_block5(feature)  # (batch_size, 256, 14, 14)
        feature = self.res_block6(feature)  # (batch_size, 256, 14, 14)

        # conv5_x
        feature = self.res_block7(feature)  # (batch_size, 512, 7, 7)
        feature = self.res_block8(feature)  # (batch_size, 512, 7, 7)

        feature = self.avg_pool(feature).view(x.size(0), -1)  # (batch_size, 512)

        score = self.linear(feature)  # (batch_size, num_classes)
        score = self.softmax(score)  # (batch_size, num_classes)

        return score
