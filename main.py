from src.dataset import BoneAgeDataset
from src.model import BoneAgeModel

dataset = BoneAgeDataset(
   csv_file="dataset/boneage-training-dataset.csv",
    image_folder="dataset/boneage-training-dataset"
)

image, label = dataset[0]

print("Image Shape:", image.shape)

# Add batch dimension
image = image.unsqueeze(0)

model = BoneAgeModel()

output = model(image)

print("Prediction Shape:", output.shape)

print("Predicted Bone Age:", output.item())