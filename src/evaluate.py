import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error

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
# LOAD DATASET
# =========================

test_dataset = BoneAgeDataset(
    csv_file="D:/Bone_age detection/dataset/val_split.csv",
    image_folder="D:/Bone_age detection/dataset/boneage-training-dataset"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# =========================
# LOAD MODEL
# =========================

model = BoneAgeModel().to(device)

model.load_state_dict(
    torch.load(
        "saved_models/best_boneage_model.pth",
        map_location=device
    )
)

model.eval()

print("Model Loaded")


# =========================
# EVALUATION
# =========================

predictions = []

actuals = []


with torch.no_grad():

    for images, genders, labels in test_loader:

        images = images.to(device)

        genders = genders.to(device)

outputs = model(images, genders)

outputs = outputs.view(-1)

predictions.extend(
    outputs.detach().cpu().numpy().tolist()
)
actuals.extend(
    labels.numpy().tolist()
)


# =========================
# CALCULATE MAE
# =========================

mae = mean_absolute_error(
    actuals,
    predictions
)

print("\n======================")
print(f"Test MAE: {mae:.2f} months")
print("======================")