import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm  # Progress bar
import albumentations as A  # Data augmentation

csv_path = r"C:\Users\chahi\Desktop\Age_bone_predection\dataset\raw\RSNA\training.csv"
images_folder = r"C:\Users\chahi\Desktop\Age_bone_predection\dataset\raw\RSNA\boneage-training-dataset"

# Load CSV
df = pd.read_csv(csv_file)
print(df.head())  # Check if it's loaded correctly

# Verify an image exists
sample_image = os.path.join(image_folder, "1377.png")  
print("Image exists:", os.path.exists(sample_image))

# Load the CSV
df = pd.read_csv(csv_path)

# Extract columns
image_ids = df["id"].astype(str)  # Convert to string
boneages = df["boneage"].values   # Bone age labels
sexes = df["male"].values         # Sex labels (0 = Female, 1 = Male)

# Lists to store preprocessed data
X_data = []
y_age = []
y_sex = []


# Detect hand orientation
def detect_hand_orientation(image):
    """Corrects hand orientation to point upwards."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 100)  # Edge detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=15)

    if lines is None:
        return image  # No adjustment needed

    angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for x1, y1, x2, y2 in lines[:, 0]]
    median_angle = np.median(angles)

    # Normalize the angle
    if median_angle < -60:
        median_angle += 90
    elif median_angle > 60:
        median_angle -= 90

    # Rotation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT)

    return rotated

# Crop hand region
def crop_hand_region(image):
    """Removes background and crops the hand."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image  # No cropping needed

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return image[y:y+h, x:x+w]

# Normalize intensity
def normalize_intensity(image):
    """Applies histogram equalization to enhance contrast."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return cv2.equalizeHist(image)

# Resize image with padding
def resize_image(image, target_size=(224, 224)):
    """Resize the image while maintaining the aspect ratio."""
    h, w = image.shape[:2]

    # Check for valid dimensions
    if h == 0 or w == 0:
        print(f"Error: Invalid image dimensions (h={h}, w={w}).")
        return None

    # Scale factor to maintain aspect ratio
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Add padding to reach target size
    delta_w = target_size[0] - new_w
    delta_h = target_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded

# Data augmentation
def apply_data_augmentation(image):
    """Applies random transformations for data augmentation."""
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ])
    return transform(image=image)["image"]

# Preprocess a single image
def preprocess_image(image_path):
    """Complete preprocessing pipeline for a single image."""
    image = cv2.imread(image_path)

    if image is None:
        print(f" Error: Unable to load {image_path}.")
        return None

    image = detect_hand_orientation(image)  # Standardize orientation
    image = crop_hand_region(image)         # Background removal
    image = normalize_intensity(image)      # Contrast enhancement
    image = resize_image(image)             # Resize with padding
    image = apply_data_augmentation(image)  # Optional data augmentation

    return np.expand_dims(image, axis=-1)   # Add channel dimension (H, W, 1)

# Process the entire dataset
for img_id, boneage, sex in tqdm(zip(image_ids, boneages, sexes), total=len(image_ids)):
    img_path = os.path.join(images_folder, img_id + ".png")

    processed_img = preprocess_image(img_path)
    if processed_img is not None:
        X_data.append(processed_img)
        y_age.append(boneage)
        y_sex.append(sex)

# Convert to NumPy arrays
X_data = np.array(X_data)
y_age = np.array(y_age)
y_sex = np.array(y_sex)

# Check final dimensions
print(f"Preprocessed images: {X_data.shape}")
print(f"Bone age labels: {y_age.shape}")
print(f"Sex labels: {y_sex.shape}")

OUTPUT_DIR = r"C:\Users\chahi\Desktop\Age_bone_predection\dataset\Processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sauvegarde des données
np.save(os.path.join(OUTPUT_DIR, "preprocessed_images.npy"), X_data)
np.save(os.path.join(OUTPUT_DIR, "labels_boneage.npy"), y_age)
np.save(os.path.join(OUTPUT_DIR, "labels_sex.npy"), y_sex)
print(" Données sauvegardées dans le répertoire npy_files !")
