import sys
import os
from pathlib import Path

# Add parent directory to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from src.dataset import BoneAgeDataset
from src.model import BoneAgeModel


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using Device:", device)


# =========================
# LOAD & VALIDATE DATA
# =========================

# Use pathlib for cross-platform paths
BASE_PATH = Path("D:/Bone_age detection/dataset")
CSV_FILE = BASE_PATH / "boneage-training-dataset.csv"
IMAGE_FOLDER = BASE_PATH / "boneage-training-dataset"

df = pd.read_csv(CSV_FILE)
print(f"Total Images in CSV: {len(df)}")

# **IMPORTANT: Filter out missing images BEFORE splitting**
missing_images = []
valid_rows = []

for idx, row in df.iterrows():
    img_path = IMAGE_FOLDER / f"{row['id']}.png"
    if not img_path.exists():
        missing_images.append(row['id'])
    else:
        valid_rows.append(idx)

if missing_images:
    print(f"\n⚠️  WARNING: {len(missing_images)} images missing from disk")
    print(f"   Missing IDs: {missing_images[:10]}...")  # Show first 10
    df = df.loc[valid_rows].reset_index(drop=True)
    print(f"   Using {len(df)} valid images\n")


# =========================
# TRAIN VALID SPLIT
# =========================

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# Save temporary CSV files
train_csv = BASE_PATH / "train_split.csv"
val_csv = BASE_PATH / "val_split.csv"

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}\n")


# =========================
# DATASETS
# =========================

train_dataset = BoneAgeDataset(
    csv_file=str(train_csv),
    image_folder=str(IMAGE_FOLDER)
)

val_dataset = BoneAgeDataset(
    csv_file=str(val_csv),
    image_folder=str(IMAGE_FOLDER)
)

print(f"Train dataset loaded: {len(train_dataset)} samples")
print(f"Val dataset loaded: {len(val_dataset)} samples\n")


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0  # Set to 0 on Windows, increase on Linux
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# =========================
# MODEL
# =========================

model = BoneAgeModel().to(device)
print(f"Model loaded on {device}\n")


# =========================
# LOSS FUNCTION & OPTIMIZER
# =========================

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# =========================
# TRAINING LOOP
# =========================

EPOCHS = 3

best_val_loss = float('inf')

for epoch in range(EPOCHS):

    # -------------------------
    # TRAINING
    # -------------------------

    model.train()

    running_loss = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_idx, (images, genders, labels) in enumerate(train_loader):

        images = images.to(device)

        genders = genders.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, genders)

        outputs = outputs.view(-1)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    if batch_idx % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Batch [{batch_idx}/{len(train_loader)}] "
            f"Train Loss: {loss.item():.4f}"
        )

    train_loss = running_loss / len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, genders, labels in val_loader:

            images = images.to(device)

            genders = genders.to(device)

            labels = labels.to(device)

            outputs = model(images, genders).squeeze()

            loss = criterion(outputs, labels)

            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    # -------------------------
    # PRINT RESULTS
    # -------------------------

    print("\n=========================")
    print(f"Epoch [{epoch+1}/{EPOCHS}] Completed")
    print(f"Training MAE: {train_loss:.4f}")
    print(f"Validation MAE: {val_loss:.4f}")
    print("=========================\n")

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
      model.state_dict(),
      "saved_models/best_boneage_model.pth"
)
          
        

        print("Best Model Saved\n")


