import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

# === Config ===
dataset_path = 'dataset/'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# === Load and Preprocess Images ===
images = []
labels = []
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)
    print(f"Loading images from class: {class_name}")
    for filename in os.listdir(class_folder):
        if filename.lower().endswith(image_extensions):
            img_path = os.path.join(class_folder, filename)
            try:
                img = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)
                images.append(np.array(img))
                labels.append(class_names.index(class_name))
            except Exception as e:
                print(f"Could not load {img_path}: {e}")

# === Convert to NumPy Arrays and Normalize ===
images = np.array(images).astype('float32') / 255.0
labels = np.array(labels)
print(f"Loaded {len(images)} images from {len(class_names)} classes.")

# === Split Dataset ===
X_trainval, X_test, y_trainval, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=42, stratify=labels)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval)

# === Noise Functions ===
def add_gaussian_noise(images, mean=0.0, std=0.05):
    noise = np.random.normal(loc=mean, scale=std, size=images.shape)
    return np.clip(images + noise, 0., 1.)

def add_salt_pepper_noise(images, amount=0.02, s_vs_p=0.5):
    noisy = images.copy()
    for i in range(len(noisy)):
        img = noisy[i]
        num_pixels = img.size
        num_salt = int(amount * num_pixels * s_vs_p)
        num_pepper = int(amount * num_pixels * (1 - s_vs_p))

        # Salt Noise
        coords = [np.random.randint(0, s, num_salt) for s in img.shape]
        img[tuple(coords)] = 1

        # Pepper Noise
        coords = [np.random.randint(0, s, num_pepper) for s in img.shape]
        img[tuple(coords)] = 0

    return noisy

# === Apply Noise to Training Data ===
X_train = add_gaussian_noise(X_train, std=0.03)
X_train = add_salt_pepper_noise(X_train, amount=0.01)

# === Convert to TensorFlow Datasets ===
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# === Define TensorFlow Data Augmentation ===
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# === Apply Augmentation to Training Set ===
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# === Define Model ===
model = keras.models.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(class_names), activation='softmax') 
])

# === Compile Model ===
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Train Model ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10 
)

# === Evaluate Model ===
test_loss, test_acc = model.evaluate(test_ds)
print(f'Test accuracy: {test_acc:.2f}')

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# === Save Model ===
model.save('micropix.h5')
