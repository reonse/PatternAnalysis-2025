# dataset_simple.py
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class OASISDataset(Dataset):
    """
    Simplest possible loader for your 6-folder layout:
    root/{train,val,test}/images + masks
    """
    def __init__(self, root="data", split="train", size=256):
        self.img_dir = Path(root) / split / "images"
        self.msk_dir = Path(root) / split / "masks"

        # find image files
        self.files = sorted([f.name for f in self.img_dir.iterdir() if f.suffix.lower().endswith(".png")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_name = self.files[i]
        mask_name = "seg_" + img_name[len("case_"):] if img_name.startswith("case_") else img_name

        img = Image.open(self.img_dir / img_name).convert("L")
        mask = Image.open(self.msk_dir / mask_name)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int32)

        # handle grayscale encodings like {0,85,170,255}
        u = np.unique(mask)
        if set(u.tolist()) != {0,1,2,3}:
            if set(u.tolist()) == {0,85,170,255}:
                lut = {0:0,85:1,170:2,255:3}
                mask = np.vectorize(lut.get)(mask).astype(np.int64)
            else:
                # generic remap
                lut = {val:i for i,val in enumerate(sorted(u.tolist()))}
                mask = np.vectorize(lut.get)(mask).astype(np.int64)

        # convert to torch tensors
        img = torch.from_numpy(img).unsqueeze(0)  # [1,H,W]
        mask = torch.from_numpy(mask.astype(np.int64))  # [H,W]
        return img, mask
