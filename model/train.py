import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model.unet import UNetResNet34
from utils.preprocessing import BrailleDataset


class DiceLoss(nn.Module):
    def forward(self, p, t):
        p = p.view(-1)
        t = t.view(-1)
        inter = (p * t).sum()
        smooth = 1.0
        return 1 - ((2 * inter + smooth) / (p.sum() + t.sum() + smooth))


def train_model():

    tr = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    ds = BrailleDataset("data/processed/", tr)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    model = UNetResNet34(pretrained=False)
    # Run on CPU (you don't have CUDA)
    device = torch.device("cpu")
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    lossf = DiceLoss()

    for ep in range(5):
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            p = model(x)
            loss = lossf(p, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("Epoch:", ep, "Loss:", loss.item())

    torch.save(model.state_dict(), "braille_unet.pth")
    print("Model saved to braille_unet.pth")
