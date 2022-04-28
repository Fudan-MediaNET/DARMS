import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_block.model import SwinTransformer


class dualchannel_net(torch.nn.Module):
    def __init__(self, csi_num, frame_len, motion_num):
        super(dualchannel_net, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.cnn1 = CNNNet(channel_num=1)
        self.cnn2 = CNNNet(channel_num=1)
        self.swin1 = SwinTransformer(in_chans=1, num_classes=motion_num)
        self.swin2 = SwinTransformer(in_chans=1, num_classes=motion_num)
        self.dense = nn.Sequential(
            nn.Linear(768*2, 128),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(128, motion_num)
        )


    def forward(self, x, y):
        x = self.cnn1(x)
        y = self.cnn2(y)
        x = self.swin1(x)
        y = self.swin2(y)
        x = torch.cat((x, y), 1)
        x = self.dense(x)
        x = F.softmax(x, dim=-1)
        return x



class CNNNet(nn.Module):
    def __init__(self, channel_num):
        super(CNNNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_num,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.02),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.02),
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.02),
        )

        self.cnn4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.02),
        )

        self.cnn5 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.02),
        )

        self.cnn6 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(negative_slope=0.02),
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)

        return x


