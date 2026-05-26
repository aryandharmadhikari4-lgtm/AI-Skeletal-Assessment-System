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
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "saved_models",
        "best_boneage_model.pth"
    )
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model Loaded Successfully")


# =========================
# IMAGE PATH
# =========================

image_path = (
    "D:/Bone_age detection/"
    "dataset/boneage-test-dataset/4440.png"
)


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
# CHRONOLOGICAL AGE
# =========================

chronological_age = 10.0  # years


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

# Convert grayscale → RGB
rgb_image = cv2.cvtColor(
    image_resized,
    cv2.COLOR_GRAY2RGB
)

rgb_image = rgb_image / 255.0

# CLAHE enhancement
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

processed_image = clahe.apply(
    image_resized
)

processed_image = (
    processed_image / 255.0
)


# =========================
# TENSOR CONVERSION
# =========================

input_tensor = torch.tensor(
    processed_image,
    dtype=torch.float32
)

# Add channel dimension
input_tensor = input_tensor.unsqueeze(0)

# Add batch dimension
input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)


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
        input_tensor,
        gender_tensor
    )

predicted_age = prediction.item()


# =========================
# AGE COMPARISON
# =========================

chronological_age_months = (
    chronological_age * 12
)

age_difference = (
    predicted_age
    - chronological_age_months
)


# =========================
# SKELETAL MATURITY STATUS
# =========================

if abs(age_difference) < 12:

    maturity_status = (
        "Normal skeletal maturation"
    )

elif age_difference <= -12:

    maturity_status = (
        "Delayed skeletal maturation"
    )

else:

    maturity_status = (
        "Advanced skeletal maturation"
    )


# =========================
# GRADCAM WRAPPER
# =========================

class GradCAMWrapper(torch.nn.Module):

    def __init__(
        self,
        model,
        gender_tensor
    ):

        super().__init__()

        self.model = model

        self.gender_tensor = gender_tensor

    def forward(self, x):

        return self.model(
            x,
            self.gender_tensor
        )


# =========================
# WRAPPED MODEL
# =========================

wrapped_model = GradCAMWrapper(
    model,
    gender_tensor
)


# =========================
# TARGET LAYER
# =========================

target_layers = [
    wrapped_model.model.cnn.layer4[-1]
]


# =========================
# GRAD-CAM
# =========================

cam = GradCAM(
    model=wrapped_model,
    target_layers=target_layers
)

grayscale_cam = cam(
    input_tensor=input_tensor
)

grayscale_cam = grayscale_cam[0]


# =========================
# HEATMAP OVERLAY
# =========================

visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)


# =========================
# SAVE OUTPUT
# =========================

OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "outputs",
        "heatmaps"
    )
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

save_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_output.png"
)

cv2.imwrite(
    save_path,
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)

print(
    f"\nHeatmap saved at: {save_path}"
)


# =========================
# AI REPORT
# =========================

print("\n========== AI REPORT ==========")

print(f"Gender: {gender_text}")

print(
    f"Predicted Bone Age: "
    f"{predicted_age:.2f} months"
)

print(
    f"Chronological Age: "
    f"{chronological_age:.1f} years"
)

print(
    f"Bone Age Difference: "
    f"{age_difference/12:.2f} years"
)

print(
    f"Skeletal Maturity Status: "
    f"{maturity_status}"
)

print("================================")


# =========================
# DISPLAY RESULTS
# =========================

plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)

plt.imshow(image, cmap='gray')

plt.title("Original X-ray")

plt.axis("off")


# Grad-CAM
plt.subplot(1, 2, 2)

plt.imshow(visualization)

plt.title(
    f"Grad-CAM\n"
    f"Gender: {gender_text}\n"
    f"Predicted Bone Age: "
    f"{predicted_age:.2f} months\n"
    f"Chronological Age: "
    f"{chronological_age:.1f} years\n"
    f"{maturity_status}"
)

plt.axis("off")

plt.tight_layout()

plt.show()