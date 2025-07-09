import os
import kagglehub
import tensorflow as tf
from PIL import Image
import numpy as np

dataset_path = 'dataset/'

# Variables
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load data
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size = IMAGE_SIZE,
    batch_size = BATCH_SIZE,
)

# Convert to tensorflow dataset
images = []
labels = []
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
class_names = sorted(os.listdir(dataset_path))

for class_name in os.listdir(dataset_path):
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
                    img.load()
                    print(f"Loaded {filename}")
                
                except Exception as e:
                    print(f"Could not load {img_dataset_path}: {e}")