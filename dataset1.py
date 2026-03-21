!pip install segment-anything transformers timm opencv-python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
class SegTextDataset(Dataset):
    def __init__(self, img_dir, mask_dir, prompt_list):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)
        self.prompts = prompt_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f" Image not found: {img_path}")
            return self.__getitem__((idx + 1) % len(self))  # skip bad sample

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 1024)) / 255.0

        mask_path = os.path.join(self.mask_dir, img_name)
        mask = cv2.imread(mask_path, 0)

        if mask is None:
            print(f" Mask not found: {mask_path}")
            return self.__getitem__((idx + 1) % len(self))
        mask = cv2.resize(mask, (1024, 1024))
        mask = (mask > 127).astype(np.float32)

        text = np.random.choice(self.prompts)
        return (
            torch.tensor(img).permute(2,0,1).float(),
            torch.tensor(mask).unsqueeze(0).float(),
            text,
            img_name
        )
