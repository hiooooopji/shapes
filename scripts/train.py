import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import json
from torch.utils.data import DataLoader, Dataset

# Paths
DATASET_PATH = "data/shape_images/"
ANNOTATION_PATH = "data/shape_annotations/"
SAVE_MODEL_PATH = "models/"

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

# Hyperparameters
IMG_SIZE = 128
LATENT_DIM = 100  # Random noise size
SHAPE_PARAMS = 5 * 4  # 5 shapes, each with 4 parameters (x, y, size, type)
EPOCHS = 999
BATCH_SIZE = 8  # Reduced batch size for faster updates
LEARNING_RATE = 0.0002

# ------------------------------
# 🎨 1. Dataset Loading
# ------------------------------
class ShapeDataset(Dataset):
    def __init__(self, img_dir, annotation_dir):
        self.img_files = sorted(os.listdir(img_dir))
        self.annotation_dir = annotation_dir
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.img_files[idx].replace(".png", ".json"))

        # Load image (grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize

        # Load shape annotations
        with open(annotation_path, "r") as f:
            shape_params = json.load(f)

        shape_vector = []
        for shape in shape_params:
            if shape["type"] == "circle":
                shape_vector.extend([shape["center"][0], shape["center"][1], shape["radius"], 0])
            elif shape["type"] == "rectangle":
                shape_vector.extend([shape["pt1"][0], shape["pt1"][1], shape["pt2"][0] - shape["pt1"][0], 1])
            elif shape["type"] == "triangle":
                shape_vector.extend([shape["points"][0][0], shape["points"][0][1], shape["points"][1][0], 2])

        while len(shape_vector) < SHAPE_PARAMS:  # Ensure fixed size
            shape_vector.append(0)

        shape_vector = torch.tensor(shape_vector, dtype=torch.float32)

        return img, shape_vector

dataset = ShapeDataset(DATASET_PATH, ANNOTATION_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------
# 🏗️ 2. Model Definitions (Generator & Discriminator)
# ------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Create models
generator = Generator(LATENT_DIM, SHAPE_PARAMS)
discriminator = Discriminator(SHAPE_PARAMS)

# Optimizers & Loss Function
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# ------------------------------
# 🚀 3. Training Loop with Progress Display
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

print(f"Training on {device}...\n")

for epoch in range(EPOCHS):
    start_time = time.time()
    num_batches = len(dataloader)
    
    for batch_idx, (real_imgs, real_shapes) in enumerate(dataloader):
        real_shapes = real_shapes.to(device)
        batch_size = real_shapes.size(0)

        # Train Discriminator
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_shapes), real_labels)

        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_shapes = generator(z)
        fake_loss = criterion(discriminator(fake_shapes.detach()), fake_labels)

        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_labels = torch.ones(batch_size, 1).to(device)  # Trick discriminator
        loss_G = criterion(discriminator(fake_shapes), fake_labels)
        loss_G.backward()
        optimizer_G.step()

        # Print progress percentage
        progress = (batch_idx + 1) / num_batches * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch {batch_idx+1}/{num_batches} ({progress:.1f}%) | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}", end="\r")
    
    # Print summary for the epoch
    epoch_time = time.time() - start_time
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] completed in {epoch_time:.2f}s | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# Save models
torch.save(generator.state_dict(), os.path.join(SAVE_MODEL_PATH, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(SAVE_MODEL_PATH, "discriminator.pth"))

print("\n✅ Training complete. Models saved!")
