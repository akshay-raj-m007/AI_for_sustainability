import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from src.dataset import LandUseDataset
from src.model import get_model
from src.train import train_model

import random
import torch

def set_seed(seed=99):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(99)

DATA_DIR = "data/raw"
RGB_DIR = os.path.join(DATA_DIR, "rgb")
RASTER_PATH = os.path.join(DATA_DIR, "worldcover_bbox_delhi_ncr_2021.tif")

# Load spatial data
ncr = gpd.read_file(os.path.join(DATA_DIR, "delhi_ncr_region.geojson"))
ncr = ncr.to_crs(epsg=4326)

# Load raster
raster = rasterio.open(RASTER_PATH)

def extract_patch(lat, lon, size=128):  #Extract 128x128 raster patch centered at (lat, lon)
    row, col = raster.index(lon, lat)
    half = size // 2
    window = Window(col-half, row-half, size, size)
    patch = raster.read(1, window=window)
    return patch

def get_dominant_class(patch):  #Return most frequent land-cover class in patch
    flat = patch.flatten()
    counts = np.bincount(flat)
    return np.argmax(counts)

# ESA mapping (3-class)
SIMPLIFIED_MAP = {
    10: "Vegetation",
    20: "Vegetation",
    30: "Vegetation",
    40: "Cropland",
    50: "Built-up"
}

class_to_idx = {
    "Cropland": 0,
    "Built-up": 1,
    "Vegetation": 2
}


# Build labeled dataset
images = os.listdir(RGB_DIR)

valid_images = []
labels = []

for img in images:
    lat, lon = img.replace(".png", "").split("_")
    lat, lon = float(lat), float(lon)
    
    patch = extract_patch(lat, lon)
    
    if patch.shape != (128,128):
        continue
    
    dominant = get_dominant_class(patch)
    
    if dominant in SIMPLIFIED_MAP:
        valid_images.append(img)
        labels.append(class_to_idx[SIMPLIFIED_MAP[dominant]])

# Train-test split (stratified)
train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    valid_images, labels, test_size=0.4,
    random_state=42, stratify=labels
)

# Data transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


# Create datasets and loaders

train_dataset = LandUseDataset(train_imgs, train_labels, RGB_DIR, train_transform)
test_dataset = LandUseDataset(test_imgs, test_labels, RGB_DIR, test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Class weights
train_counts = Counter(train_labels)
total_samples = sum(train_counts.values())

class_weights = torch.tensor(
    [total_samples/train_counts[i] for i in range(3)],
    dtype=torch.float
)

model = get_model()

model, train_losses, test_losses, train_accs, test_accs, test_f1s = train_model(
    model,
    train_loader,
    test_loader,
    epochs=10,
    class_weights=class_weights
)

torch.save(model.state_dict(), "outputs/model.pth")


epochs_range = range(1, len(train_losses)+1)

plt.figure(figsize=(12,5))

# Loss curve
plt.subplot(1,2,1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# Accuracy curve
plt.subplot(1,2,2)
plt.plot(epochs_range, train_accs, label="Train Accuracy")
plt.plot(epochs_range, test_accs, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_curves.png")
plt.show()