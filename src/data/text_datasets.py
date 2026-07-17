"""Datasets/utilities for controllable-text training and shortcut evaluation."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.data.dataset import find_image_path


def build_counterfactual_df(df, label_col="label"):
    """Duplicate each row into a concordant-text copy and a flipped-text copy,
    BOTH carrying the true label. Requires `text_concordant` / `text_flipped`
    columns. The model thus sees identical image+label with contradictory leading
    text, so leading language becomes uninformative and it must use image+facts.
    """
    a = df.copy(); a["text"] = a["text_concordant"]
    b = df.copy(); b["text"] = b["text_flipped"]
    out = pd.concat([a, b], ignore_index=False).sort_index(kind="stable")
    return out.reset_index(drop=True)


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
