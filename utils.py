import numpy as np
import cv2
import torch

def crop_word(image, coords):
    pts = np.array(coords, dtype=np.int32).reshape(4, 2)

    x, y, w, h = cv2.boundingRect(pts)

    crop = image[y:y+h, x:x+w]
    return crop
def parse_line(line):
    parts = line.strip().split(',')
    coords = list(map(int, parts[:8]))
    text = ",".join(parts[8:])
    return coords, text

def encode(text, char_to_idx):
    seq = [char_to_idx[c] for c in text if c in char_to_idx]
    if len(seq) == 0:
        return None
    return torch.tensor(seq)