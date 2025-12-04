import os
import cv2
from torch.utils.data import Dataset


class BrailleDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.items = [f for f in os.listdir(folder) if f.endswith(".png")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name = self.items[idx]

        img = cv2.imread(os.path.join(self.folder, img_name), 0)
        mask = cv2.imread(os.path.join(self.folder, img_name.replace(".png", "_mask.png")), 0)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
