import torch
import torch.nn as nn
from torchvision.models import resnet34


class UNetResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        base = resnet34(pretrained=pretrained)

        # Encoder
        self.e0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.e1 = nn.Sequential(base.maxpool, base.layer1)
        self.e2 = base.layer2
        self.e3 = base.layer3
        self.e4 = base.layer4

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.c4 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        d1 = self.up1(e4)
        d1 = self.c1(torch.cat([d1, e3], dim=1))

        d2 = self.up2(d1)
        d2 = self.c2(torch.cat([d2, e2], dim=1))

        d3 = self.up3(d2)
        d3 = self.c3(torch.cat([d3, e1], dim=1))

        d4 = self.up4(d3)
        d4 = self.c4(torch.cat([d4, e0], dim=1))

        return torch.sigmoid(self.out(d4))
