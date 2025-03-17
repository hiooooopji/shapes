import torch
import torch.nn as nn
import numpy as np
import cv2

# Define the Generator
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

# Hyperparameters
LATENT_DIM = 100
SHAPE_PARAMS = 5 * 4  # Output shape size
IMG_SIZE = 128  # Image size

# Load the trained Generator
generator = Generator(LATENT_DIM, SHAPE_PARAMS)
generator.load_state_dict(torch.load("models/generator.pth", map_location=torch.device("cpu")))
generator.eval()

# Generate a new shape vector
z = torch.randn(1, LATENT_DIM)  # Random noise input
generated_shape = generator(z).detach().numpy().flatten()

# Create a blank image
image = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255

# Interpret the shape vector and draw shapes
for i in range(5):  # 5 shapes
    x = int(generated_shape[i * 4] % IMG_SIZE)
    y = int(generated_shape[i * 4 + 1] % IMG_SIZE)
    size = int(abs(generated_shape[i * 4 + 2]) % (IMG_SIZE // 4))
    shape_type = int(abs(generated_shape[i * 4 + 3]) % 3)  # 0 = circle, 1 = rect, 2 = triangle

    if shape_type == 0:  # Draw circle
        cv2.circle(image, (x, y), size, (0), -1)
    elif shape_type == 1:  # Draw rectangle
        cv2.rectangle(image, (x, y), (x + size, y + size), (0), -1)
    elif shape_type == 2:  # Draw triangle
        pts = np.array([[x, y], [x + size, y + size], [x - size, y + size]], np.int32)
        cv2.fillPoly(image, [pts], (0))

# Save and open the image
output_path = "generated_image.png"
cv2.imwrite(output_path, image)
print(f"✅ Image saved as '{output_path}'. Open it manually to view.")
