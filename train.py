import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


dataset_path = 'dataset/'

# Variables
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load data
images = []
labels = []
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
class_names = sorted(os.listdir(dataset_path))

for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)
    
    if os.path.isdir(class_folder):
        print(f"Loading images from class: {class_name}")
        
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(image_extensions):
                img_dataset_path = os.path.join(class_folder, filename)
                
                try:
                    img = Image.open(img_dataset_path).convert('RGB')
                    img = img.resize(IMAGE_SIZE)
                    images.append(np.array(img))
                    labels.append(class_names.index(class_name))
                except Exception as e:
                    print(f"Could not load {img_dataset_path}: {e}")
                    
# Convert to numpy arrays     
images = np.array(images)
labels = np.array(labels)

print(f"Loaded {len(images)} images from {len(class_names)} classes.")

# Normalize
images = images.astype('float32') / 255.0

# Test Split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=42, stratify=labels
)

# Training and Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)