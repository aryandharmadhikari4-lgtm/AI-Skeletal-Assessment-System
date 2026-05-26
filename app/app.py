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
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.model import BoneAgeModel


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Skeletal Assessment System",
    layout="wide"
)

st.title("AI Skeletal Assessment System")

st.write(
    """
    Upload a hand X-ray to perform:
    - Bone age prediction
    - Skeletal maturity analysis
    - Growth plate assessment
    - Comparative bone growth analysis
    - Grad-CAM explainability
    """
)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    model = BoneAgeModel().to(device)

    MODEL_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
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

    return model


model = load_model()


# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Hand X-ray",
    type=["png", "jpg", "jpeg"]
)


# =========================
# GENDER INPUT
# =========================

gender_option = st.radio(
    "Select Gender",
    ["Male", "Female"]
)

gender_value = (
    1 if gender_option == "Male"
    else 0
)


# =========================
# CHRONOLOGICAL AGE INPUT
# =========================

chronological_age = st.number_input(
    "Enter Chronological Age (years)",
    min_value=0.0,
    max_value=25.0,
    value=10.0,
    step=0.1
)


# =========================
# PROCESS IMAGE
# =========================

if uploaded_file is not None:

    # =========================
    # LOAD IMAGE
    # =========================

    image = Image.open(uploaded_file)

    image = np.array(image)

    # Convert RGB to grayscale
    if len(image.shape) == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    original_image = image.copy()

    # =========================
    # PREPROCESSING
    # =========================

    image_resized = cv2.resize(
        image,
        (224, 224)
    )

    # RGB image for Grad-CAM
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

    predicted_years = (
        predicted_age / 12
    )

    # =========================
    # CONFIDENCE INTERVAL
    # =========================

    model_mae = 7.86

    lower_bound = (
        predicted_age - model_mae
    )

    upper_bound = (
        predicted_age + model_mae
    )

    # =========================
    # CONFIDENCE LEVEL
    # =========================

    if model_mae <= 8:

        confidence_level = "High"

    elif model_mae <= 12:

        confidence_level = "Moderate"

    else:

        confidence_level = "Low"

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

    age_gap_years = (
        age_difference / 12
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
    # DEVELOPMENT STAGE
    # =========================

    if predicted_years < 2:

        development_stage = (
            "Infant"
        )

    elif predicted_years < 6:

        development_stage = (
            "Early Childhood"
        )

    elif predicted_years < 10:

        development_stage = (
            "Childhood"
        )

    elif predicted_years < 13:

        development_stage = (
            "Pre-pubertal"
        )

    elif predicted_years < 16:

        development_stage = (
            "Pubertal"
        )

    elif predicted_years < 18:

        development_stage = (
            "Late Pubertal"
        )

    else:

        development_stage = (
            "Near Skeletal Maturity"
        )

    # =========================
    # GROWTH PLATE STATUS
    # =========================

    if predicted_years < 10:

        growth_plate_status = (
            "Widely open growth plates"
        )

    elif predicted_years < 13:

        growth_plate_status = (
            "Open growth plates"
        )

    elif predicted_years < 15:

        growth_plate_status = (
            "Partially fused growth plates"
        )

    elif predicted_years < 17:

        growth_plate_status = (
            "Advanced epiphyseal fusion"
        )

    else:

        growth_plate_status = (
            "Mostly fused growth plates"
        )

    # =========================
    # COMPARATIVE ANALYSIS
    # =========================

    if abs(age_gap_years) < 0.5:

        comparative_analysis = (
            "Skeletal development is "
            "consistent with chronological age."
        )

    elif age_gap_years < -0.5:

        comparative_analysis = (
            f"Skeletal development appears "
            f"{abs(age_gap_years):.1f} years "
            f"behind expected maturation."
        )

    else:

        comparative_analysis = (
            f"Skeletal development appears "
            f"{age_gap_years:.1f} years "
            f"ahead of expected maturation."
        )

    # =========================
    # MATURITY SCORE
    # =========================

    maturity_score = (
        predicted_years / 18
    ) * 100

    maturity_score = min(
        maturity_score,
        100
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

    wrapped_model = GradCAMWrapper(
        model,
        gender_tensor
    )

    # =========================
    # TARGET LAYERS
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

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # =========================
    # DISPLAY RESULTS
    # =========================

    st.subheader("AI Skeletal Assessment Report")

    st.write(
        f"Predicted Bone Age: "
        f"{predicted_age:.2f} months"
    )

    st.write(
        f"Estimated Prediction Range: "
        f"{lower_bound:.2f} - "
        f"{upper_bound:.2f} months"
    )

    st.write(
        f"Prediction Confidence: "
        f"{confidence_level}"
    )

    st.write(
        f"Chronological Age: "
        f"{chronological_age:.1f} years"
    )

    st.write(
        f"Bone Age Difference: "
        f"{age_gap_years:.2f} years"
    )

    st.write(
        f"Skeletal Maturity Status: "
        f"{maturity_status}"
    )

    st.write(
        f"Development Stage: "
        f"{development_stage}"
    )

    st.write(
        f"Growth Plate Status: "
        f"{growth_plate_status}"
    )

    st.write(
        f"Skeletal Maturity Score: "
        f"{maturity_score:.1f}%"
    )

    st.write(
        "Comparative Skeletal Analysis:"
    )

    st.info(
        comparative_analysis
    )

    # =========================
    # SKELETAL MATURITY BAR
    # =========================

    st.write("Skeletal Maturity Progress")

    st.progress(
        min(
            max(
                predicted_years / 18,
                0
            ),
            1
        )
    )

    # =========================
    # COMPARISON CHART
    # =========================

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    categories = [
        "Chronological Age",
        "Predicted Bone Age"
    ]

    values = [
        chronological_age,
        predicted_years
    ]

    ax.bar(categories, values)

    ax.set_ylabel("Age (Years)")

    ax.set_title(
        "Comparative Bone Growth Analysis"
    )

    st.pyplot(fig)

    # =========================
    # DISPLAY IMAGES
    # =========================

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            original_image,
            caption="Original X-ray",
            use_container_width=True
        )

    with col2:

        st.image(
            visualization,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )