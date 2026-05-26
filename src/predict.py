import sys
import os

# Add parent directory
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import cv2
import torch
import matplotlib.pyplot as plt

from src.model import BoneAgeModel


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


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

print("Model Loaded Successfully")


# =========================
# IMAGE PATH
# =========================

image_path = (
    "D:/XRay_dataset/boneage-training-dataset/boneage-training-dataset/4599.png"
)

# Change image if needed


# =========================
# GENDER INPUT
# =========================

# Male = 1
# Female = 0

gender_value = 1

gender_text = (
    "Male"
    if gender_value == 1
    else "Female"
)


# =========================
# LOAD IMAGE
# =========================

image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if image is None:

    raise ValueError(
        f"Failed to load image: {image_path}"
    )


# =========================
# PREPROCESSING
# =========================

image_resized = cv2.resize(
    image,
    (224, 224)
)

# CLAHE enhancement
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

image_resized = clahe.apply(
    image_resized
)

image_resized = image_resized / 255.0


# =========================
# TENSOR CONVERSION
# =========================

image_tensor = torch.tensor(
    image_resized,
    dtype=torch.float32
)

# Add channel dimension
image_tensor = image_tensor.unsqueeze(0)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# =========================
# GENDER TENSOR
# =========================

gender_tensor = torch.tensor(
    [gender_value],
    dtype=torch.float32
).to(device)


# =========================
# PREDICTION
# =========================

with torch.no_grad():

    prediction = model(
        image_tensor,
        gender_tensor
    )

predicted_age = prediction.item()


# =========================
# OUTPUT
# =========================

print(
    f"\nGender: {gender_text}"
)

print(
    f"Predicted Bone Age: "
    f"{predicted_age:.2f} months"
)


# =========================
# DISPLAY IMAGE
# =========================

plt.figure(figsize=(6, 6))

plt.imshow(image, cmap='gray')

plt.title(
    f"{gender_text}\n"
    f"Predicted Bone Age: "
    f"{predicted_age:.2f} months"
)

plt.axis("off")

plt.show()