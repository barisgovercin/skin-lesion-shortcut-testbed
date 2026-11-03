"""Datasets/utilities for controllable-text training and shortcut evaluation."""
from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import find_image_path
# Lives in the torch-free text module so the CPU demo can use it; re-exported here.
from src.preprocessing.synthetic_text import build_counterfactual_df  # noqa: F401


class PairedTextDataset(torch.utils.data.Dataset):
    """Yields (image, ids_c, mask_c, ids_f, mask_f, label) using the concordant and
    flipped tokenised text - for SRS (flip test) and the consistency loss."""

    def __init__(self, df, img_dirs, tokenizer, transform=None, max_length=128):
        self.df = df.reset_index(drop=True)
        self.img_dirs = [Path(d) for d in img_dirs]
        self.tok = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _img(self, image_id):
        return find_image_path(self.img_dirs, image_id)

    def _tok(self, text):
        e = self.tok(str(text), padding="max_length", truncation=True,
                     max_length=self.max_length, return_tensors="pt")
        return e["input_ids"].squeeze(0), e["attention_mask"].squeeze(0)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self._img(r["image_id"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        ic, mc = self._tok(r["text_concordant"])
        iff, mf = self._tok(r["text_flipped"])
        return img, ic, mc, iff, mf, torch.tensor(r["label"], dtype=torch.float32)
