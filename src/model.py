import torch
import torch.nn as nn
import torchvision.models as models


class BoneAgeModel(nn.Module):

    def __init__(self):

        super(BoneAgeModel, self).__init__()

        # =========================
        # RESNET BACKBONE
        # =========================

        self.cnn = models.resnet18(
            pretrained=True
        )

        # Modify first layer for grayscale
        self.cnn.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Remove final classification layer
        self.cnn.fc = nn.Identity()

        # =========================
        # REGRESSION HEAD
        # =========================

        self.regressor = nn.Sequential(

            nn.Linear(512 + 1, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, image, gender):

        # CNN feature extraction
        features = self.cnn(image)

        # Reshape gender
        gender = gender.unsqueeze(1)

        # Concatenate image features + gender
        combined = torch.cat(
            (features, gender),
            dim=1
        )

        # Final prediction
        output = self.regressor(combined)

        return output