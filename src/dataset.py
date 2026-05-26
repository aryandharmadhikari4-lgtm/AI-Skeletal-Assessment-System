import os
import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset


class BoneAgeDataset(Dataset):

    def __init__(self, csv_file, image_folder):

        self.df = pd.read_csv(csv_file)

        self.image_folder = image_folder

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        # =========================
        # GET ROW
        # =========================

        row = self.df.iloc[idx]

        # =========================
        # IMAGE ID COLUMN
        # =========================

        if 'id' in row:
            img_id = int(row['id'])

        elif 'Case ID' in row:
            img_id = int(row['Case ID'])

        else:
            raise KeyError(
                "No image ID column found"
            )

        # =========================
        # BONE AGE COLUMN
        # =========================

        if 'boneage' in row:
            bone_age = row['boneage']

        elif 'bone_age' in row:
            bone_age = row['bone_age']

        elif 'Bone Age' in row:
            bone_age = row['Bone Age']

        elif 'Ground truth bone age (months)' in row:
            bone_age = row[
                'Ground truth bone age (months)'
            ]

        else:
            raise KeyError(
                "No bone age column found"
            )

        # =========================
        # GENDER
        # =========================

        if 'male' in row:

            gender = 1 if row['male'] == True else 0

        elif 'Sex' in row:

            gender = 1 if row['Sex'] == 'M' else 0

        else:

            gender = 0

        # =========================
        # IMAGE PATH
        # =========================

        png_path = os.path.join(
            self.image_folder,
            f"{img_id}.png"
        )

        jpg_path = os.path.join(
            self.image_folder,
            f"{img_id}.jpg"
        )

        jpeg_path = os.path.join(
            self.image_folder,
            f"{img_id}.jpeg"
        )

        if os.path.exists(png_path):

            img_path = png_path

        elif os.path.exists(jpg_path):

            img_path = jpg_path

        elif os.path.exists(jpeg_path):

            img_path = jpeg_path

        else:

            raise FileNotFoundError(
                f"No image found for ID {img_id}"
            )

        # =========================
        # LOAD IMAGE
        # =========================

        image = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise ValueError(
                f"Failed to load image: {img_path}"
            )

        # =========================
        # PREPROCESSING
        # =========================

        image = cv2.resize(
            image,
            (224, 224)
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        image = clahe.apply(image)

        image = image / 255.0

        # =========================
        # TENSOR CONVERSION
        # =========================

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        image = image.unsqueeze(0)

        gender = torch.tensor(
            gender,
            dtype=torch.float32
        )

        bone_age = torch.tensor(
            bone_age,
            dtype=torch.float32
        )

        # =========================
        # RETURN
        # =========================

        return image, gender, bone_age