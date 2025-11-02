from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class OASISDataset(Dataset):
    """
    Loads 2D grayscale images and integer label masks.
    Expects 6-folder layout:
      root/{train,val,test}/{images,masks}/*.png
    Maps `case_XXX_slice_Y.nii.png -> seg_XXX_slice_Y.nii.png`.
    If identical names are used, that works too.
    """
    def __init__(self, root: str = "data", split: str = "train"):
        self.img_dir = Path(root) / split / "images"
        self.msk_dir = Path(root) / split / "masks"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.img_dir}")
        if not self.msk_dir.exists():
            raise FileNotFoundError(f"Masks dir not found: {self.msk_dir}")

        self.files = sorted([f.name for f in self.img_dir.iterdir() if f.suffix.lower().endswith(".png")])
        if len(self.files) == 0:
            raise RuntimeError(f"No PNG images found in {self.img_dir}")

    @staticmethod
    def _default_mask_name(img_name: str) -> str:
        if img_name.startswith("case_"):
            return "seg_" + img_name[len("case_"):]
        return img_name

    @staticmethod
    def _remap_mask(mask_np: np.ndarray) -> np.ndarray:
        """
        Map mask intensities to {0,1,2,3}.
        Handles grayscale palettes like {0,85,170,255}.
        """
        u = np.unique(mask_np)
        s = set(u.tolist())
        if s <= {0,1,2,3}:
            return mask_np.astype(np.int64)
        if s == {0,85,170,255}:
            lut = {0:0,85:1,170:2,255:3}
            return np.vectorize(lut.get)(mask_np).astype(np.int64)
        # generic fallback
        lut = {val:i for i,val in enumerate(sorted(u.tolist()))}
        return np.vectorize(lut.get)(mask_np).astype(np.int64)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        img_name = self.files[idx]
        msk_name = self._default_mask_name(img_name)
        img_path = self.img_dir / img_name
        msk_path = self.msk_dir / msk_name

        if not msk_path.exists():
            alt = self.msk_dir / img_name
            if alt.exists():
                msk_path = alt
            else:
                raise FileNotFoundError(f"Mask not found for {img_name}")

        img = Image.open(img_path).convert("L")
        msk = Image.open(msk_path)

        img = np.array(img, dtype=np.float32) / 255.0
        msk = np.array(msk, dtype=np.int32)
        msk = self._remap_mask(msk)

        x = torch.from_numpy(img).unsqueeze(0).contiguous()  # [1,H,W]
        y = torch.from_numpy(msk.astype(np.int64)).contiguous()
        return x, y
