"""Dataset classes and data utilities for skin lesion classification."""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Very large images can trip PIL's DecompressionBomb guard (a warning above
# ~89 MP and a hard error above ~179 MP). These are our own trusted local files,
# so lift the cap to avoid log spam and a mid-run crash on an oversized image.
Image.MAX_IMAGE_PIXELS = None


IMG_EXTS = (".jpg", ".jpeg", ".png")


def find_image_path(img_dirs, image_id, exts=IMG_EXTS):
    """Locate an image by id across candidate dirs, trying common extensions.

    Datasets may store '.jpg' (matched first), '.jpeg', or '.png'.
    """
    for d in img_dirs:
        d = Path(d)
        for ext in exts:
            p = d / f"{image_id}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(
        f"Image not found: {image_id} (dirs={[str(x) for x in img_dirs]})")


class SkinLesionDataset(Dataset):
    """Vision-only skin lesion dataset."""

    def __init__(self, df, img_dirs, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dirs = [Path(d) for d in img_dirs]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image(self, image_id):
        return find_image_path(self.img_dirs, image_id)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self._find_image(row["image_id"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label


class MultimodalSkinDataset(Dataset):
    """Multimodal dataset: image + patient metadata."""

    def __init__(self, df, img_dirs, transform=None, metadata_cols=None):
        self.df = df.reset_index(drop=True)
        self.img_dirs = [Path(d) for d in img_dirs]
        self.transform = transform
        self.metadata_cols = metadata_cols or []

    def __len__(self):
        return len(self.df)

    def _find_image(self, image_id):
        return find_image_path(self.img_dirs, image_id)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self._find_image(row["image_id"])).convert("RGB")
        if self.transform:
            img = self.transform(img)

        metadata = torch.tensor(
            [row[col] for col in self.metadata_cols], dtype=torch.float32
        )
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, metadata, label


class MultimodalTextDataset(Dataset):
    """Image + tokenized clinical text, with optional tabular metadata.

    Returns batches matching the trainer's text modes:
      - metadata_cols is None  -> (image, input_ids, attention_mask, label)
      - metadata_cols is set   -> (image, metadata, input_ids, attention_mask, label)

    Text is read from `text_col` (added beforehand by add_synthetic_text) and
    tokenized to a fixed length so the default collate stacks cleanly.
    """

    def __init__(self, df, img_dirs, tokenizer, transform=None,
                 text_col="text", metadata_cols=None, max_length=128):
        self.df = df.reset_index(drop=True)
        self.img_dirs = [Path(d) for d in img_dirs]
        self.tokenizer = tokenizer
        self.transform = transform
        self.text_col = text_col
        self.metadata_cols = metadata_cols or []
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _find_image(self, image_id):
        return find_image_path(self.img_dirs, image_id)

    def _tokenize(self, text):
        enc = self.tokenizer(
            str(text),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self._find_image(row["image_id"])).convert("RGB")
        if self.transform:
            img = self.transform(img)

        input_ids, attention_mask = self._tokenize(row[self.text_col])
        label = torch.tensor(row["label"], dtype=torch.float32)

        if self.metadata_cols:
            metadata = torch.tensor(
                [row[col] for col in self.metadata_cols], dtype=torch.float32
            )
            return img, metadata, input_ids, attention_mask, label
        return img, input_ids, attention_mask, label


def get_transforms(split, img_size=224):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def encode_metadata_ham10000(df, engineered=False):
    """Encode HAM10000 metadata as numerical features.

    Features: age (normalized), sex (binary), localization (one-hot).
    Following the paper's approach of encoding tabular data for the model.
    """
    result = df.copy()

    # Age: normalize to [0, 1]
    result["age_norm"] = result["age"].fillna(result["age"].median()) / 100.0

    # Sex: binary encoding
    result["sex_male"] = (result["sex"] == "male").astype(float)

    # Localization: one-hot top categories, rest as "other"
    top_locations = [
        "back", "lower extremity", "trunk", "upper extremity",
        "abdomen", "face", "chest", "foot", "neck", "scalp",
    ]
    for loc in top_locations:
        result[f"loc_{loc}"] = (result["localization"] == loc).astype(float)

    metadata_cols = (
        ["age_norm", "sex_male"]
        + [f"loc_{loc}" for loc in top_locations]
    )

    if engineered:
        from src.data.metadata_features import add_engineered, HAM_UV_MAP
        metadata_cols = metadata_cols + add_engineered(
            result, "localization", "age", HAM_UV_MAP
        )

    return result, metadata_cols


def prepare_ham10000(data_dir="data/raw/ham10000", seed=42, include_metadata=False, engineered=False):
    """Prepare HAM10000 with 60/20/20 per-lesion stratified split."""
    meta = pd.read_csv(Path(data_dir) / "HAM10000_metadata.csv")

    malignant_classes = ["mel", "bcc", "akiec"]
    meta["label"] = meta["dx"].apply(lambda x: 1 if x in malignant_classes else 0)

    # Per-lesion stratified split (no leakage)
    lesion_df = meta.groupby("lesion_id").first().reset_index()

    train_lesions, temp_lesions = train_test_split(
        lesion_df["lesion_id"],
        test_size=0.4,
        stratify=lesion_df["label"],
        random_state=seed,
    )
    temp_df = lesion_df[lesion_df["lesion_id"].isin(temp_lesions)]
    val_lesions, test_lesions = train_test_split(
        temp_df["lesion_id"],
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=seed,
    )

    train_df = meta[meta["lesion_id"].isin(train_lesions)]
    val_df = meta[meta["lesion_id"].isin(val_lesions)]
    test_df = meta[meta["lesion_id"].isin(test_lesions)]

    img_dirs = [
        Path(data_dir) / "HAM10000_images_part_1",
        Path(data_dir) / "HAM10000_images_part_2",
    ]

    metadata_cols = None
    if include_metadata:
        train_df, metadata_cols = encode_metadata_ham10000(train_df, engineered=engineered)
        val_df, _ = encode_metadata_ham10000(val_df, engineered=engineered)
        test_df, _ = encode_metadata_ham10000(test_df, engineered=engineered)

    print(f"Train: {len(train_df)} ({train_df['label'].mean():.1%} malignant)")
    print(f"Val:   {len(val_df)} ({val_df['label'].mean():.1%} malignant)")
    print(f"Test:  {len(test_df)} ({test_df['label'].mean():.1%} malignant)")
    if metadata_cols:
        print(f"Metadata features: {len(metadata_cols)} ({metadata_cols})")

    return train_df, val_df, test_df, img_dirs, metadata_cols


def stratified_group_split(df, group_col, label_col, seed=42, test_size=0.4, val_test_ratio=0.5):
    """Split into train/val/test with NO group leakage.

    Groups (e.g. patients) are kept entirely within one split. Stratification
    uses a group-level label (a group is positive if any of its rows is
    positive), which approximately preserves the malignant ratio across splits.
    Returns (train_df, val_df, test_df).
    """
    group_label = df.groupby(group_col)[label_col].max()
    groups = group_label.index.to_numpy()
    glabels = group_label.to_numpy()

    train_groups, temp_groups = train_test_split(
        groups, test_size=test_size, stratify=glabels, random_state=seed,
    )
    temp_labels = group_label.loc[temp_groups].to_numpy()
    val_groups, test_groups = train_test_split(
        temp_groups, test_size=val_test_ratio, stratify=temp_labels, random_state=seed,
    )

    train_df = df[df[group_col].isin(train_groups)]
    val_df = df[df[group_col].isin(val_groups)]
    test_df = df[df[group_col].isin(test_groups)]
    return train_df, val_df, test_df


def encode_metadata_isic2020(df, engineered=False, patient_aggregates=False):
    """Encode ISIC 2020 metadata as numerical features.

    Features: age_approx (normalized), sex (binary), anatomical site (one-hot).
    Mirrors encode_metadata_ham10000 so the multimodal model is dataset-agnostic.
    """
    result = df.copy()

    median_age = result["age_approx"].median()
    fill_age = median_age if pd.notna(median_age) else 0.0
    result["age_norm"] = result["age_approx"].fillna(fill_age) / 100.0

    result["sex_male"] = (result["sex"] == "male").astype(float)

    sites = [
        "head/neck", "upper extremity", "lower extremity",
        "torso", "palms/soles", "oral/genital",
    ]

    def site_col(s):
        return "site_" + s.replace("/", "_").replace(" ", "_")

    for s in sites:
        result[site_col(s)] = (result["anatom_site_general_challenge"] == s).astype(float)

    metadata_cols = ["age_norm", "sex_male"] + [site_col(s) for s in sites]

    if engineered:
        from src.data.metadata_features import add_engineered, ISIC_UV_MAP
        metadata_cols = metadata_cols + add_engineered(
            result, "anatom_site_general_challenge", "age_approx", ISIC_UV_MAP
        )

    if patient_aggregates and "patient_id" in result.columns:
        counts = result["patient_id"].map(result["patient_id"].value_counts())
        result["images_per_patient"] = counts.astype(float)
        result["log_images_per_patient"] = np.log1p(counts.astype(float))
        metadata_cols = metadata_cols + ["images_per_patient", "log_images_per_patient"]

    return result, metadata_cols


def prepare_isic2020(data_dir="data/raw/isic2020", seed=42, include_metadata=False,
                     engineered=False, patient_aggregates=False):
    """Prepare ISIC 2020 with 60/20/20 per-patient stratified split (no patient leakage)."""
    data_dir = Path(data_dir)

    csv_candidates = [
        "train.csv",
        "train_concat.csv",
        "ISIC_2020_Training_GroundTruth_v2.csv",
        "ISIC_2020_Training_GroundTruth.csv",
    ]
    csv_path = next((data_dir / c for c in csv_candidates if (data_dir / c).exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"No ISIC 2020 metadata CSV found in {data_dir}. Tried: {csv_candidates}"
        )
    meta = pd.read_csv(csv_path)

    # Standardise the image-id column so the Dataset classes work unchanged.
    if "image_name" in meta.columns:
        meta = meta.rename(columns={"image_name": "image_id"})

    # Binary label: prefer the explicit 'target' column, else derive it.
    if "target" in meta.columns:
        meta["label"] = meta["target"].astype(int)
    elif "benign_malignant" in meta.columns:
        meta["label"] = (meta["benign_malignant"] == "malignant").astype(int)
    else:
        raise ValueError("ISIC CSV missing both 'target' and 'benign_malignant' columns.")

    # Rows without a patient_id become their own singleton group (no false grouping).
    if "patient_id" not in meta.columns:
        meta["patient_id"] = [f"_nopatient_{i}" for i in range(len(meta))]
    else:
        missing = meta["patient_id"].isna()
        if missing.any():
            meta.loc[missing, "patient_id"] = [f"_nopatient_{i}" for i in range(int(missing.sum()))]

    train_df, val_df, test_df = stratified_group_split(
        meta, group_col="patient_id", label_col="label", seed=seed,
    )

    # Candidate image dirs — _find_image checks each, so unknown layouts are tolerated.
    img_dirs = [
        data_dir / "train" / "train",
        data_dir / "train",
        data_dir / "jpeg" / "train",
        data_dir / "256x256",
        data_dir,
    ]

    metadata_cols = None
    if include_metadata:
        train_df, metadata_cols = encode_metadata_isic2020(
            train_df, engineered=engineered, patient_aggregates=patient_aggregates)
        val_df, _ = encode_metadata_isic2020(
            val_df, engineered=engineered, patient_aggregates=patient_aggregates)
        test_df, _ = encode_metadata_isic2020(
            test_df, engineered=engineered, patient_aggregates=patient_aggregates)

    n_patients = meta["patient_id"].nunique()
    print(f"ISIC 2020: {len(meta)} images, {n_patients} patient groups")
    print(f"Train: {len(train_df)} ({train_df['label'].mean():.1%} malignant)")
    print(f"Val:   {len(val_df)} ({val_df['label'].mean():.1%} malignant)")
    print(f"Test:  {len(test_df)} ({test_df['label'].mean():.1%} malignant)")
    if metadata_cols:
        print(f"Metadata features: {len(metadata_cols)} ({metadata_cols})")

    return train_df, val_df, test_df, img_dirs, metadata_cols


ISIC_DICM_SITES = [
    "lower extremity", "anterior torso", "head/neck", "posterior torso",
    "upper extremity", "palms/soles", "lateral torso", "oral/genital",
]


def encode_metadata_isic_dicm_17k(df, engineered=False):
    """Encode ISIC-DICM-17K metadata (Ahammed et al., 2025) as numerical features.

    Columns: age_approx, sex, anatom_site_general. Mirrors the ISIC 2020 encoder
    but uses this dataset's finer torso vocabulary (anterior/posterior/lateral).
    """
    result = df.copy()

    median_age = result["age_approx"].median()
    fill_age = median_age if pd.notna(median_age) else 0.0
    result["age_norm"] = result["age_approx"].fillna(fill_age) / 100.0

    result["sex_male"] = (result["sex"] == "male").astype(float)

    def site_col(s):
        return "site_" + s.replace("/", "_").replace(" ", "_")

    for s in ISIC_DICM_SITES:
        result[site_col(s)] = (result["anatom_site_general"] == s).astype(float)

    metadata_cols = ["age_norm", "sex_male"] + [site_col(s) for s in ISIC_DICM_SITES]

    if engineered:
        from src.data.metadata_features import add_engineered, ISIC_UV_MAP
        metadata_cols = metadata_cols + add_engineered(
            result, "anatom_site_general", "age_approx", ISIC_UV_MAP
        )

    return result, metadata_cols


def prepare_isic_dicm_17k(data_dir="data/raw/isic_dicm_17k", seed=42,
                          include_metadata=False, engineered=False):
    """Prepare ISIC-DICM-17K with a 60/20/20 per-lesion stratified split.

    The repo ships fixed train/valid CSVs; we pool them and re-split per seed
    (grouped by lesion_id, singleton fallback) so the 5-seed CI protocol and the
    no-leakage guarantee match the rest of the thesis. Dataset is class-balanced
    (8,530 / 8,530), so class weights are ~1.
    """
    data_dir = Path(data_dir)

    frames = []
    for name in ("train-set-metadata.csv", "valid-set-metadata.csv"):
        p = data_dir / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(
            f"No ISIC-DICM-17K metadata CSVs in {data_dir}. "
            f"Run: python scripts/download_isic_dicm_17k.py"
        )
    meta = pd.concat(frames, ignore_index=True)

    meta = meta.rename(columns={"isic_id": "image_id"})
    meta["label"] = meta["class"].astype(int)

    # Group by lesion to avoid leakage; rows without a lesion_id become singletons.
    if "lesion_id" not in meta.columns:
        meta["lesion_id"] = np.nan
    missing = meta["lesion_id"].isna()
    meta.loc[missing, "lesion_id"] = [f"_solo_{iid}" for iid in meta.loc[missing, "image_id"]]

    train_df, val_df, test_df = stratified_group_split(
        meta, group_col="lesion_id", label_col="label", seed=seed,
    )

    img_dirs = [data_dir / "images", data_dir]

    metadata_cols = None
    if include_metadata:
        train_df, metadata_cols = encode_metadata_isic_dicm_17k(train_df, engineered=engineered)
        val_df, _ = encode_metadata_isic_dicm_17k(val_df, engineered=engineered)
        test_df, _ = encode_metadata_isic_dicm_17k(test_df, engineered=engineered)

    print(f"ISIC-DICM-17K: {len(meta)} images, {meta['lesion_id'].nunique()} lesion groups")
    print(f"Train: {len(train_df)} ({train_df['label'].mean():.1%} malignant)")
    print(f"Val:   {len(val_df)} ({val_df['label'].mean():.1%} malignant)")
    print(f"Test:  {len(test_df)} ({test_df['label'].mean():.1%} malignant)")
    if metadata_cols:
        print(f"Metadata features: {len(metadata_cols)} ({metadata_cols})")

    return train_df, val_df, test_df, img_dirs, metadata_cols


def compute_class_weights(df):
    """Compute class weights: w_i = N / (2 * n_i) following the paper."""
    n = len(df)
    n_pos = df["label"].sum()
    n_neg = n - n_pos
    w_neg = n / (2 * n_neg)
    w_pos = n / (2 * n_pos)
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)

