import os
import json
import cv2
import numpy as np
import random

# Folder setup
DATASET_SIZE = 1000  # Number of images to generate
IMG_SIZE = 128  # Image size (128x128)
DATASET_PATH = "data/shape_images/"
ANNOTATION_PATH = "data/shape_annotations/"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(ANNOTATION_PATH, exist_ok=True)

# Shape drawing function
def draw_random_shape(img, shape_type, color, thickness=2):
    h, w, _ = img.shape
    center = (random.randint(20, w-20), random.randint(20, h-20))
    size = random.randint(10, 40)

    if shape_type == "circle":
        cv2.circle(img, center, size, color, thickness)
        return {"type": "circle", "center": center, "radius": size}
    elif shape_type == "rectangle":
        pt1 = (center[0] - size, center[1] - size)
        pt2 = (center[0] + size, center[1] + size)
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return {"type": "rectangle", "pt1": pt1, "pt2": pt2}
    elif shape_type == "triangle":
        pt1 = (center[0], center[1] - size)
        pt2 = (center[0] - size, center[1] + size)
        pt3 = (center[0] + size, center[1] + size)
        cv2.polylines(img, [np.array([pt1, pt2, pt3])], isClosed=True, color=color, thickness=thickness)
        return {"type": "triangle", "points": [pt1, pt2, pt3]}

# Generate dataset
for i in range(DATASET_SIZE):
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    num_shapes = random.randint(1, 3)

    shapes_info = []
    for _ in range(num_shapes):
        shape_type = random.choice(["circle", "rectangle", "triangle"])
        color = (0, 0, 0)  # Black shapes
        shape_info = draw_random_shape(img, shape_type, color)
        shapes_info.append(shape_info)

    # Save image
    img_path = os.path.join(DATASET_PATH, f"image_{i}.png")
    cv2.imwrite(img_path, img)

    # Save annotations
    annotation_path = os.path.join(ANNOTATION_PATH, f"image_{i}.json")
    with open(annotation_path, "w") as f:
        json.dump(shapes_info, f)

print(f"✅ Dataset of {DATASET_SIZE} shape-based images created!")
