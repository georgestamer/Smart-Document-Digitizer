# Smart Document Digitizer

An OCR pipeline that reads word crops from grocery receipt images and transcribes them using a CRNN (Convolutional Recurrent Neural Network) trained with CTC loss.

## Overview

The pipeline:
1. Downloads the [SROIE 2019 dataset](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) via `kagglehub`
2. Parses bounding-box annotations and crops each word from the receipt image
3. Feeds grayscale word crops through a CRNN model
4. Trains with CTC loss to align predicted character sequences with ground-truth text

## Architecture

**CRNN** (`CRNN.py`)
- 2-layer CNN (Conv → ReLU → MaxPool) extracts spatial features from each word crop
- Bidirectional 2-layer LSTM reads the feature sequence left-to-right and right-to-left
- Linear head outputs per-timestep class logits over the character vocabulary

**Dataset** (`dataset.py` — `SROIEDataset`)
- Loads images, box files, and entity files from the SROIE dataset directory structure
- Crops and grayscale-converts each annotated word region, resized to `128×32`
- Returns `(image_tensor, text)` pairs

**Utilities** (`utils.py`)
- `parse_line` — parses a box-file line into 8 coordinates + text label
- `crop_word` — crops a quadrilateral region from an image using its bounding rect
- `encode` — converts a string to a tensor of character indices

## Dataset

[SROIE 2019](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) — scanned grocery receipts with word-level bounding boxes and key-entity annotations (company, date, address, total).

Expected directory layout (downloaded automatically):
```
SROIE2019/
  train/
    img/        # receipt images
    box/        # word bounding-box annotations
    entities/   # key-value entity JSON files
```

## Character Vocabulary

```
# <space> A-Z a-z 0-9 . , : - /
```

Index `0` is reserved as the CTC blank token.

## Getting Started

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run training**

Open and run `main.ipynb`. The notebook will:
- Download the dataset via `kagglehub`
- Build the dataset and data loader
- Train the CRNN for the configured number of epochs
- Print per-epoch CTC loss

**Hyperparameters** (set in `main.ipynb`)

| Parameter | Default |
|---|---|
| Epochs | 1 |
| Learning rate | 1e-3 |
| Batch size | 8 |
| Image size | 128 × 32 |
| LSTM hidden size | 256 |

## Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- OpenCV, Pillow, NumPy
- kagglehub, tqdm

See `requirements.txt` for the full list.
