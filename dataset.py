import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

class SROIEDataset(Dataset):
    def __init__(self, img_dir, box_dir, entity_dir, img_size=(128, 32)):
        self.img_dir = img_dir
        self.box_dir = box_dir
        self.entity_dir = entity_dir
        self.img_size = img_size

        # IMPORTANT: enforce deterministic ordering
        self.img_files = sorted(os.listdir(img_dir))
        self.box_files = sorted(os.listdir(box_dir))
        self.entity_files = sorted(os.listdir(entity_dir))

        assert len(self.img_files) == len(self.box_files) == len(self.entity_files), \
            "Mismatch between img/box/entity counts"

        self.samples = []

        for i in range(len(self.img_files)):
            img_path = os.path.join(img_dir, self.img_files[i])
            box_path = os.path.join(box_dir, self.box_files[i])
            ent_path = os.path.join(entity_dir, self.entity_files[i])

            image = cv2.imread(img_path)

            # optional: read entity file (not used in OCR training here)
            with open(ent_path, "r", encoding="utf-8") as f:
                entity_lines = f.readlines()

            with open(box_path, "r", encoding="utf-8") as f:
                box_lines = f.readlines()

            for line in box_lines:
                coords, text = parse_line(line)

                crop = crop_word(image, coords)

                if crop is None or crop.size == 0:
                    continue

                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop = cv2.resize(crop, self.img_size)

                self.samples.append((crop, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, text = self.samples[idx]

        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.unsqueeze(0)  # (1, H, W)

        return img, text