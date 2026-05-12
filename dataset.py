import torch
from torch.utils.data import Dataset
import cv2
import os
from utils import parse_line, crop_word

class SROIEDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        for file in os.listdir(root):
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                txt_path = img_path.replace(".jpg", ".txt")

                image = cv2.imread(img_path)

                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    coords, text = parse_line(line)

                    crop = crop_word(image, coords)
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    crop = cv2.resize(crop, (128, 32))

                    self.samples.append((crop, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, text = self.samples[idx]

        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.unsqueeze(0)  # (1, H, W)

        return img, text