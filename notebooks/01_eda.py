"""
Exploratory Data Analysis - HAM10000 & ISIC 2020
Skin Cancer Detection MSc Thesis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import random

sns.set_theme(style="whitegrid", font_scale=1.1)
random.seed(42)

# %% [markdown]
# ## 1. HAM10000 Dataset

# %%
ham_meta = pd.read_csv("../data/raw/ham10000/HAM10000_metadata.csv")
print(f"HAM10000: {len(ham_meta)} records, {ham_meta['lesion_id'].nunique()} unique lesions")
print(f"Columns: {list(ham_meta.columns)}")
ham_meta.head()

# %%
malignant_classes = ["mel", "bcc", "akiec"]
ham_meta["label"] = ham_meta["dx"].apply(lambda x: 1 if x in malignant_classes else 0)
ham_meta["label_name"] = ham_meta["label"].map({0: "Benign", 1: "Malignant"})

print("=== Binary Class Distribution ===")
print(ham_meta["label_name"].value_counts())
print(f"\nMalignant ratio: {ham_meta['label'].mean():.1%}")

# %%
dx_names = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratosis",
    "vasc": "Vascular lesion",
    "df": "Dermatofibroma",
}
ham_meta["dx_full"] = ham_meta["dx"].map(dx_names)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

order = ham_meta["dx_full"].value_counts().index
colors = ["#e74c3c" if dx in ["Melanoma", "Basal cell carcinoma", "Actinic keratosis"] else "#3498db" for dx in order]
sns.countplot(data=ham_meta, y="dx_full", order=order, hue="dx_full", palette=dict(zip(order, colors)), legend=False, ax=axes[0])
axes[0].set_title("HAM10000 - 7-Class Distribution")
axes[0].set_xlabel("Number of Images")
axes[0].set_ylabel("")
for i, v in enumerate(ham_meta["dx_full"].value_counts()[order].values):
    axes[0].text(v + 30, i, str(v), va="center", fontweight="bold")

sns.countplot(data=ham_meta, x="label_name", hue="label_name", palette={"Benign": "#3498db", "Malignant": "#e74c3c"}, legend=False, ax=axes[1])
axes[1].set_title("HAM10000 - Binary Distribution")
axes[1].set_xlabel("")
axes[1].set_ylabel("Number of Images")
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("../results/ham10000_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ham_meta.dropna(subset=["age"]).groupby("label_name")["age"].plot.kde(ax=axes[0], legend=True)
axes[0].set_title("Age Distribution (Benign vs Malignant)")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Density")

sex_label = ham_meta.groupby(["sex", "label_name"]).size().unstack(fill_value=0)
sex_label.plot(kind="bar", ax=axes[1], color=["#3498db", "#e74c3c"])
axes[1].set_title("Sex vs Class Distribution")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

top_loc = ham_meta["localization"].value_counts().head(8).index
loc_data = ham_meta[ham_meta["localization"].isin(top_loc)]
loc_label = loc_data.groupby(["localization", "label_name"]).size().unstack(fill_value=0)
loc_label = loc_label.loc[loc_label.sum(axis=1).sort_values(ascending=True).index]
loc_label.plot(kind="barh", ax=axes[2], color=["#3498db", "#e74c3c"])
axes[2].set_title("Localization vs Class Distribution (Top 8)")
axes[2].set_xlabel("Number of Images")
axes[2].set_ylabel("")

plt.tight_layout()
plt.savefig("../results/ham10000_demographics.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(2, 7, figsize=(20, 6))
fig.suptitle("HAM10000 - Sample Images per Class", fontsize=14, fontweight="bold")

img_dirs = [
    Path("../data/raw/ham10000/HAM10000_images_part_1"),
    Path("../data/raw/ham10000/HAM10000_images_part_2"),
]

for col_idx, dx in enumerate(["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]):
    samples = ham_meta[ham_meta["dx"] == dx].sample(2, random_state=42)
    for row_idx, (_, row) in enumerate(samples.iterrows()):
        img_path = None
        for d in img_dirs:
            p = d / f"{row['image_id']}.jpg"
            if p.exists():
                img_path = p
                break
        if img_path:
            img = Image.open(img_path).resize((224, 224))
            axes[row_idx, col_idx].imshow(img)
        label = "MAL" if dx in malignant_classes else "BEN"
        color = "red" if dx in malignant_classes else "blue"
        axes[row_idx, col_idx].set_title(f"{dx_names[dx]}\n({label})", fontsize=8, color=color)
        axes[row_idx, col_idx].axis("off")

plt.tight_layout()
plt.savefig("../results/ham10000_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 2. ISIC 2020 Dataset

# %%
isic_meta = pd.read_csv("../data/raw/isic2020/train_concat.csv")
print(f"ISIC 2020: {len(isic_meta)} records, {isic_meta['patient_id'].nunique()} unique patients")
print(f"Columns: {list(isic_meta.columns)}")
isic_meta["label_name"] = isic_meta["target"].map({0: "Benign", 1: "Malignant"})
isic_meta.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(data=isic_meta, x="label_name", hue="label_name", palette={"Benign": "#3498db", "Malignant": "#e74c3c"}, legend=False, ax=axes[0])
axes[0].set_title("ISIC 2020 - Binary Distribution")
axes[0].set_xlabel("")
axes[0].set_ylabel("Number of Images")
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontweight='bold')

imgs_per_patient = isic_meta.groupby("patient_id").size()
axes[1].hist(imgs_per_patient.values, bins=50, color="#2ecc71", edgecolor="black")
axes[1].set_title("Images per Patient")
axes[1].set_xlabel("Number of Images")
axes[1].set_ylabel("Number of Patients")
axes[1].axvline(imgs_per_patient.median(), color="red", linestyle="--", label=f"Median: {imgs_per_patient.median():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig("../results/isic2020_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

isic_meta.dropna(subset=["age_approx"]).groupby("label_name")["age_approx"].plot.kde(ax=axes[0], legend=True)
axes[0].set_title("Age Distribution (Benign vs Malignant)")
axes[0].set_xlabel("Age")

sex_label = isic_meta.dropna(subset=["sex"]).groupby(["sex", "label_name"]).size().unstack(fill_value=0)
sex_label.plot(kind="bar", ax=axes[1], color=["#3498db", "#e74c3c"])
axes[1].set_title("Sex vs Class Distribution")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

loc_label = isic_meta.dropna(subset=["anatom_site_general_challenge"])
top_sites = loc_label["anatom_site_general_challenge"].value_counts().head(6).index
loc_filtered = loc_label[loc_label["anatom_site_general_challenge"].isin(top_sites)]
site_counts = loc_filtered.groupby(["anatom_site_general_challenge", "label_name"]).size().unstack(fill_value=0)
site_counts = site_counts.loc[site_counts.sum(axis=1).sort_values(ascending=True).index]
site_counts.plot(kind="barh", ax=axes[2], color=["#3498db", "#e74c3c"])
axes[2].set_title("Anatomical Site vs Class Distribution")
axes[2].set_xlabel("Number of Images")
axes[2].set_ylabel("")

plt.tight_layout()
plt.savefig("../results/isic2020_demographics.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.suptitle("ISIC 2020 - Sample Images (Top: Benign, Bottom: Malignant)", fontsize=14, fontweight="bold")

img_dir = Path("../data/raw/isic2020/train/train")

for row_idx, target in enumerate([0, 1]):
    samples = isic_meta[isic_meta["target"] == target].sample(5, random_state=42)
    for col_idx, (_, row) in enumerate(samples.iterrows()):
        img_path = img_dir / f"{row['image_name']}.jpg"
        if img_path.exists():
            img = Image.open(img_path).resize((224, 224))
            axes[row_idx, col_idx].imshow(img)
        label = "Malignant" if target == 1 else "Benign"
        color = "red" if target == 1 else "blue"
        age = f"Age:{int(row['age_approx'])}" if pd.notna(row['age_approx']) else "Age:?"
        sex = row['sex'] if pd.notna(row['sex']) else "?"
        axes[row_idx, col_idx].set_title(f"{label}\n{age}, {sex}", fontsize=9, color=color)
        axes[row_idx, col_idx].axis("off")

plt.tight_layout()
plt.savefig("../results/isic2020_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 3. Dataset Comparison

# %%
comparison = pd.DataFrame({
    "HAM10000": [
        len(ham_meta),
        ham_meta["lesion_id"].nunique(),
        "7-class -> binary",
        f"{ham_meta['label'].mean():.1%}",
        f"{ham_meta['age'].mean():.1f} +/- {ham_meta['age'].std():.1f}",
        f"{(ham_meta['sex']=='female').mean():.1%}",
        "localization (15 regions)",
        "N/A",
    ],
    "ISIC 2020": [
        len(isic_meta),
        isic_meta["patient_id"].nunique(),
        "Binary (0/1)",
        f"{isic_meta['target'].mean():.1%}",
        f"{isic_meta['age_approx'].mean():.1f} +/- {isic_meta['age_approx'].std():.1f}",
        f"{(isic_meta['sex']=='female').mean():.1%}",
        "anatom_site (9 regions)",
        "patient_id (for per-patient split)",
    ],
}, index=[
    "Total images",
    "Unique lesions/patients",
    "Class structure",
    "Malignant ratio",
    "Mean age",
    "Female ratio",
    "Localization",
    "Patient ID",
])

print("=== Dataset Comparison ===")
print(comparison.to_string())

# %%
print("\n=== Image Size Analysis ===")

ham_sizes = []
for d in img_dirs:
    for img_path in list(d.glob("*.jpg"))[:50]:
        img = Image.open(img_path)
        ham_sizes.append(img.size)

isic_sizes = []
for img_path in list(img_dir.glob("*.jpg"))[:100]:
    img = Image.open(img_path)
    isic_sizes.append(img.size)

ham_sizes = np.array(ham_sizes)
isic_sizes = np.array(isic_sizes)

print(f"HAM10000  - Avg: {ham_sizes[:,0].mean():.0f}x{ham_sizes[:,1].mean():.0f}, "
      f"Min: {ham_sizes[:,0].min()}x{ham_sizes[:,1].min()}, Max: {ham_sizes[:,0].max()}x{ham_sizes[:,1].max()}")
print(f"ISIC 2020 - Avg: {isic_sizes[:,0].mean():.0f}x{isic_sizes[:,1].mean():.0f}, "
      f"Min: {isic_sizes[:,0].min()}x{isic_sizes[:,1].min()}, Max: {isic_sizes[:,0].max()}x{isic_sizes[:,1].max()}")

print("\nEDA complete! Results saved to results/ directory.")
