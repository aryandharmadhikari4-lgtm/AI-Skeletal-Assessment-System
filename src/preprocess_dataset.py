import os
import pandas as pd

# Paths
csv_path = "../dataset/boneage-training-dataset.csv"
image_folder = "../dataset/boneage-training-dataset"
cleaned_csv_path = "../dataset/boneage-training-dataset-cleaned.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Verify image paths
valid_rows = []
for _, row in df.iterrows():
    img_id = row['id']
    img_path = os.path.join(image_folder, f"{img_id}.png")
    if os.path.exists(img_path):
        valid_rows.append(row)
    else:
        print(f"Missing image: {img_path}")

# Create cleaned DataFrame
cleaned_df = pd.DataFrame(valid_rows)

# Save cleaned CSV
cleaned_df.to_csv(cleaned_csv_path, index=False)
print(f"Cleaned dataset saved to {cleaned_csv_path}")